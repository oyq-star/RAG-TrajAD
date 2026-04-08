"""
Training script for RAG-TrajAD v12.
Stage 1: Self-supervised pretraining (non-target domains only)
Stage 2: Multi-source normal-only memory (ALL non-target domains)
Stage 3: Scorer training (frozen encoder, multi-source data, CrossAttention)
Evaluation: Auto-selects best of scorer/kNN/blend scoring
"""

import os
import sys
import json
import math
import random
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

sys.path.insert(0, str(Path(__file__).parent))
from dataset import get_dataset, TrajectoryDataset
from model import RAGTrajAD, CrossDomainMemory

DOMAIN_NAMES = ['porto', 'tdrive', 'geolife']


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k == 'subtype':
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def pad_batch(batch):
    """Pad variable-length trajectories."""
    max_len = max(b['cell_tokens'].shape[0] for b in batch)
    for b in batch:
        for tok_key in ['cell_tokens', 'motion_tokens', 'time_tokens']:
            t = b[tok_key]
            if t.shape[0] < max_len:
                b[tok_key] = F.pad(t, (0, max_len - t.shape[0]))
    return collate_fn(batch)


# ─────────────────────────────────────────────
# Stage 1: Self-supervised pretraining
# ─────────────────────────────────────────────

def pretrain(model, source_datasets, args, device):
    """Masked token reconstruction + optional domain adversarial pretraining."""
    no_adv = getattr(args, 'no_domain_adv', False)
    print(f"=== Stage 1: Self-supervised pretraining (domain_adv={'OFF' if no_adv else 'ON'}) ===")
    combined = ConcatDataset(source_datasets)
    loader = DataLoader(combined, batch_size=args.pretrain_batch_size,
                        shuffle=True, num_workers=2, collate_fn=pad_batch,
                        drop_last=True)

    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=args.pretrain_lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.pretrain_epochs * len(loader))

    model.encoder.train()
    for epoch in range(args.pretrain_epochs):
        p = epoch / max(args.pretrain_epochs - 1, 1)
        if not no_adv:
            grl_alpha = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0
            grl_max = getattr(args, 'grl_max_weight', 0.5)
            dom_weight = grl_max * (0.2 + 0.8 * p)
            model.encoder.domain_classifier[0].set_alpha(grl_alpha)

        total_loss, total_recon, total_dom = 0.0, 0.0, 0.0
        for step, batch in enumerate(loader):
            cell = batch['cell_tokens'].to(device)
            motion = batch['motion_tokens'].to(device)
            time = batch['time_tokens'].to(device)
            domain_ids = batch['domain_id'].to(device)
            B, T = cell.shape

            mask = torch.rand(B, T, device=device) < model.encoder.mask_prob
            out = model.encoder(cell, motion, time, mask=mask)

            cell_loss = F.cross_entropy(
                out['cell_logits'][mask], cell[mask], ignore_index=0)
            motion_loss = F.cross_entropy(
                out['motion_logits'][mask], motion[mask], ignore_index=0)
            recon_loss = cell_loss + motion_loss

            if no_adv:
                loss = recon_loss
                dom_loss_val = 0.0
            else:
                dom_loss = F.cross_entropy(out['domain_logits'], domain_ids)
                loss = recon_loss + dom_weight * dom_loss
                dom_loss_val = dom_loss.item()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_dom += dom_loss_val

        avg = total_loss / len(loader)
        print(f"  Epoch {epoch+1}/{args.pretrain_epochs}: loss={avg:.4f} "
              f"recon={total_recon/len(loader):.4f} dom={total_dom/len(loader):.4f}")

    return model


# ─────────────────────────────────────────────
# Stage 2: Normal-only memory construction
# ─────────────────────────────────────────────

@torch.no_grad()
def build_memory(model, source_datasets, args, device):
    """Build normal-only memory from source data. No anomalies, no KMeans."""
    print("=== Stage 2: Building normal-only memory ===")
    memory = CrossDomainMemory(d_model=args.d_model, slot_len=args.slot_len)
    model.encoder.eval()

    n_normal = 0
    n_skipped = 0

    for ds in source_datasets:
        mem_bs = getattr(args, 'memory_batch_size', 256)
        loader = DataLoader(ds, batch_size=mem_bs, shuffle=False,
                            num_workers=2, collate_fn=pad_batch)
        for batch in loader:
            cell = batch['cell_tokens'].to(device)
            motion = batch['motion_tokens'].to(device)
            time = batch['time_tokens'].to(device)
            labels = batch['label']
            domains = batch['domain_id']

            out = model.encoder(cell, motion, time, embed_only=True)
            z_norm = out['z_norm']
            token_feats = out['token_features']

            for i in range(len(labels)):
                if labels[i].item() == 0:  # NORMAL ONLY
                    memory.add(
                        z_norm=z_norm[i].cpu(),
                        token_features=token_feats[i].cpu(),
                        label=0,
                        domain_id=domains[i].item(),
                        subtype_name=f"dom{domains[i].item()}",
                        subtype_id=domains[i].item(),
                    )
                    n_normal += 1
                else:
                    n_skipped += 1

    memory.build_index()

    print(f"  Memory: {n_normal} normal entries (skipped {n_skipped} anomalies)")
    return memory


# ─────────────────────────────────────────────
# Stage 3: End-to-end scorer training
# ─────────────────────────────────────────────

def train_scorer(model, source_datasets, memory, target_domain_id, args, device):
    """Frozen encoder + CrossAttentionScorer on multi-source normal slots."""
    print(f"=== Stage 3: Scorer training (frozen encoder, multi-source, target: {DOMAIN_NAMES[target_domain_id]}) ===")

    train_ds = [ds for ds in source_datasets if ds.DOMAIN_ID != target_domain_id]
    combined = ConcatDataset(train_ds)
    loader = DataLoader(combined, batch_size=args.scorer_batch_size,
                        shuffle=True, num_workers=2, collate_fn=pad_batch,
                        drop_last=True)

    key_matrix = memory.key_matrix.to(device)
    slot_matrix = memory.slot_matrix.to(device)
    domain_tensor = memory.domain_tensor.to(device)

    # Only train scorer parameters (encoder is frozen)
    optimizer = torch.optim.AdamW(model.scorer.parameters(), lr=args.scorer_lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.scorer_epochs * len(loader))

    model.encoder.eval()  # FROZEN
    model.scorer.train()

    top_k = model.top_k
    sl = slot_matrix.shape[1]  # slot_len
    d = slot_matrix.shape[2]

    for epoch in range(args.scorer_epochs):
        total_loss = 0.0
        for batch in loader:
            cell = batch['cell_tokens'].to(device)
            motion = batch['motion_tokens'].to(device)
            time_tok = batch['time_tokens'].to(device)
            y = batch['label'].float().to(device)

            with torch.no_grad():
                out = model.encoder(cell, motion, time_tok, embed_only=True)
                z = out['z']
                z_norm = out['z_norm']

                cos_sim = z_norm @ key_matrix.T  # (B, M)
                domain_mask = (domain_tensor == target_domain_id)
                cos_sim = cos_sim.masked_fill(domain_mask.unsqueeze(0), -1e9)
                k = min(top_k, key_matrix.shape[0])
                _, topk_idx = cos_sim.topk(k, dim=1)  # (B, K)

                z_neighbors = key_matrix[topk_idx]  # (B, K, d)
                B = z_norm.shape[0]
                idx_exp = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, sl, d)
                slots_normal = slot_matrix.unsqueeze(0).expand(B, -1, -1, -1).gather(1, idx_exp)

            score = model.scorer(z, z_norm, slots_normal, z_neighbors)
            loss = F.binary_cross_entropy_with_logits(score, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.scorer.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        print(f"  Epoch {epoch+1}/{args.scorer_epochs}: "
              f"scorer_loss={total_loss/len(loader):.4f}")

    return model


# ─────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser("RAG-TrajAD v7 Training")
    # Data
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--source', type=str, required=True,
                        help='Source domain name: porto|tdrive|geolife')
    parser.add_argument('--target', type=str, required=True,
                        help='Target domain name: porto|tdrive|geolife')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--detour_threshold', type=float, default=1.5)

    # Model
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_encoder_layers', type=int, default=4)
    parser.add_argument('--n_query_slots', type=int, default=8)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--slot_len', type=int, default=8)

    # Pretraining
    parser.add_argument('--pretrain_epochs', type=int, default=10)
    parser.add_argument('--pretrain_batch_size', type=int, default=128)
    parser.add_argument('--pretrain_lr', type=float, default=1e-3)
    parser.add_argument('--no_domain_adv', action='store_true',
                        help='Disable domain adversarial training (ablation A4)')
    parser.add_argument('--grl_max_weight', type=float, default=0.5,
                        help='Maximum domain adversarial loss weight (default 0.5)')

    # Scorer
    parser.add_argument('--scorer_epochs', type=int, default=15)
    parser.add_argument('--scorer_batch_size', type=int, default=64)
    parser.add_argument('--memory_batch_size', type=int, default=256)
    parser.add_argument('--scorer_lr', type=float, default=1e-4)

    # K-shot
    parser.add_argument('--k_shot', type=int, default=0,
                        help='Number of target anomaly labels for scorer fine-tuning')

    # Output
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--exp_name', type=str, default='default')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

    return parser.parse_args()


def auto_batch_sizes(args):
    """Auto-set batch sizes based on available GPU memory."""
    if not torch.cuda.is_available():
        return
    total_mem = torch.cuda.get_device_properties(0).total_memory
    mem_gb = total_mem / 1024**3
    # Try nvidia-smi for free memory (works on NVIDIA GPUs)
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            smi_free_mb = int(result.stdout.strip().split('\n')[0])
            mem_gb = smi_free_mb / 1024
    except Exception:
        pass
    if args.pretrain_batch_size == 128:
        args.pretrain_batch_size = min(256, max(32, int((mem_gb - 3) / 0.1)))
    if args.scorer_batch_size == 64:
        args.scorer_batch_size = min(512, max(64, int((mem_gb - 2) / 0.02)))
    args.memory_batch_size = min(1024, max(128, int((mem_gb - 1) / 0.005)))
    print(f"  Auto batch sizes (GPU ~{mem_gb:.0f}GB): pretrain={args.pretrain_batch_size}, "
          f"scorer={args.scorer_batch_size}, memory={args.memory_batch_size}")


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, Source: {args.source} -> Target: {args.target}, K={args.k_shot}")

    auto_batch_sizes(args)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load ALL non-target domains for pretraining (multi-source DG)
    pretrain_domain_names = [n for n in DOMAIN_NAMES if n != args.target]
    pretrain_sets = []
    for name in pretrain_domain_names:
        try:
            ds = get_dataset(name, args.data_path, split='train',
                             max_len=args.max_len, seed=args.seed,
                             detour_threshold=args.detour_threshold)
            pretrain_sets.append(ds)
            print(f"  Loaded {name} train (pretrain): {len(ds)} samples")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    # Use ALL non-target domains for stages 2-3 (multi-source DG)
    source_train_sets = pretrain_sets  # all non-target domains
    print(f"  Multi-source train (stages 2-3): {sum(len(d) for d in source_train_sets)} samples "
          f"from {[DOMAIN_NAMES[d.DOMAIN_ID] for d in source_train_sets]}")

    target_test_ds = get_dataset(args.target, args.data_path, split='test',
                                  max_len=args.max_len, seed=args.seed,
                                  detour_threshold=args.detour_threshold)
    print(f"  Target {args.target} test: {len(target_test_ds)}")

    target_domain_id = {'porto': 0, 'tdrive': 1, 'geolife': 2}[args.target]

    # Build model (v7: normal-only + CrossAttention)
    n_domains = len(DOMAIN_NAMES)
    model = RAGTrajAD(
        d_model=args.d_model, n_heads=args.n_heads,
        n_encoder_layers=args.n_encoder_layers,
        n_query_slots=args.n_query_slots,
        n_domains=n_domains, top_k=args.top_k,
    ).to(device)

    # Stage 1: Pretrain — reuse encoder checkpoint if available, else train fresh
    no_adv = getattr(args, 'no_domain_adv', False)
    grl_max = getattr(args, 'grl_max_weight', 0.5)
    if no_adv:
        ckpt_prefix = "noadv_encoder"
    elif grl_max != 0.5:
        ckpt_prefix = f"encoder_grl{grl_max}"
    else:
        ckpt_prefix = "encoder"
    v5_ckpt = os.path.join(args.checkpoint_dir, f"{ckpt_prefix}_notarget_{args.target}.pt")
    if os.path.exists(v5_ckpt):
        print(f"  Loading encoder from {v5_ckpt}")
        model.encoder.load_state_dict(torch.load(v5_ckpt, map_location=device))
    else:
        model = pretrain(model, pretrain_sets, args, device)
        torch.save(model.encoder.state_dict(), v5_ckpt)

    # Stage 2: Build normal-only memory
    memory = build_memory(model, source_train_sets, args, device)

    # Stage 3: Scorer training (frozen encoder, cross-attention on normal slots)
    model = train_scorer(model, source_train_sets, memory, target_domain_id, args, device)

    # ─── Evaluation ───
    print("=== Evaluation ===")
    args._target_domain_id = target_domain_id
    from evaluate import evaluate_model
    results = evaluate_model(model, memory, target_test_ds, args, device)
    results.update({
        'source': args.source,
        'target': args.target,
        'k_shot': args.k_shot,
        'seed': args.seed,
        'exp_name': args.exp_name,
    })

    out_path = os.path.join(args.output_dir, f"{args.exp_name}.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")
    print(f"  AUROC={results['auroc']:.4f}  AUPRC={results['auprc']:.4f}  "
          f"FPR@95TPR={results['fpr95']:.4f}")

    return results


if __name__ == '__main__':
    main()
