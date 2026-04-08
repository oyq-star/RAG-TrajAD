"""
Baseline implementations for RAG-TrajAD comparison.
Baselines: IBOAT, t2vec+kNN, Deep SAD, DANN+Transformer, ProtoNet,
           Source-only Transformer, Target-only oracle, AdapTime+kNN.
"""

import os
import sys
import json
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.neighbors import LocalOutlierFactor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dataset import (get_dataset, haversine_m, trajectory_length_m,
                     shortest_path_approx_m, CELL_VOCAB_SIZE, MOTION_BINS, TIME_BINS)
from model import TrajectoryEncoder, GradientReversal
from train import set_seed, pad_batch, DOMAIN_NAMES
from evaluate import compute_fpr_at_tpr


# ─────────────────────────────────────────────
# Common evaluation helper
# ─────────────────────────────────────────────

def eval_scores(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    if len(np.unique(y_true)) < 2:
        return {'auroc': 0.5, 'auprc': float(y_true.mean()), 'fpr95': 1.0}
    return {
        'auroc': float(roc_auc_score(y_true, y_score)),
        'auprc': float(average_precision_score(y_true, y_score)),
        'fpr95': float(compute_fpr_at_tpr(y_true, y_score)),
    }


# ─────────────────────────────────────────────
# Baseline 1: IBOAT (simplified)
# Isolation-Based Online Anomalous Trajectory Detection
# Uses trajectory feature distances — simplified version with LOF
# ─────────────────────────────────────────────

def extract_traj_features(samples):
    """Hand-crafted trajectory features for classical baselines."""
    features = []
    labels = []
    for coords, timestamps, label, subtype in samples:
        if len(coords) < 3:
            continue
        traj_len = trajectory_length_m(coords)
        sp_len = shortest_path_approx_m(coords)
        duration = max(timestamps[-1] - timestamps[0], 1.0)
        avg_speed = traj_len / duration
        detour_ratio = traj_len / max(sp_len, 1.0)
        n_pts = len(coords)

        # Heading variance
        headings = []
        for i in range(1, len(coords)):
            dy = coords[i][0] - coords[i-1][0]
            dx = math.cos(math.radians(coords[i-1][0])) * (coords[i][1] - coords[i-1][1])
            headings.append(math.atan2(dx, dy))
        heading_var = float(np.var(headings)) if headings else 0.0

        features.append([traj_len, sp_len, duration, avg_speed,
                         detour_ratio, n_pts, heading_var])
        labels.append(label)
    return np.array(features), np.array(labels)


def run_iboat(source_datasets, target_test_ds, args):
    """IBOAT baseline using LOF on hand-crafted features."""
    # Fit on source normal data
    source_feats, source_labels = [], []
    for ds in source_datasets:
        f, l = extract_traj_features(ds.samples)
        source_feats.append(f)
        source_labels.append(l)
    source_feats = np.concatenate(source_feats)
    source_labels = np.concatenate(source_labels)
    normal_feats = source_feats[source_labels == 0]

    # Normalize
    mean, std = normal_feats.mean(0), normal_feats.std(0) + 1e-8
    normal_feats_n = (normal_feats - mean) / std

    lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
    lof.fit(normal_feats_n)

    # Score target
    test_feats, test_labels = extract_traj_features(target_test_ds.samples)
    test_feats_n = (test_feats - mean) / std
    scores = -lof.score_samples(test_feats_n)  # higher = more anomalous
    return eval_scores(test_labels, scores)


# ─────────────────────────────────────────────
# Baseline 2: t2vec + kNN
# Pre-trained trajectory representation + kNN anomaly scoring
# ─────────────────────────────────────────────

def run_t2vec_knn(source_datasets, target_test_ds, args, device):
    """t2vec proxy: train autoencoder on source, kNN scoring on target."""
    d = args.d_model
    encoder = TrajectoryEncoder(d_model=d, n_heads=4, n_layers=2,
                                dropout=0.1, n_domains=3).to(device)
    # Quick pretraining (autoencoder)
    combined = ConcatDataset(source_datasets)
    loader = DataLoader(combined, batch_size=128, shuffle=True,
                        collate_fn=pad_batch, drop_last=True)
    opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    encoder.train()
    for epoch in range(3):
        for batch in loader:
            cell = batch['cell_tokens'].to(device)
            motion = batch['motion_tokens'].to(device)
            time = batch['time_tokens'].to(device)
            mask = torch.rand_like(cell.float()) < 0.15
            out = encoder(cell, motion, time, mask=mask)
            loss = (F.cross_entropy(out['cell_logits'][mask], cell[mask]) +
                    F.cross_entropy(out['motion_logits'][mask], motion[mask]))
            opt.zero_grad(); loss.backward(); opt.step()

    # Encode source normals
    encoder.eval()
    source_z_normals = []
    with torch.no_grad():
        for ds in source_datasets:
            ld = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=pad_batch)
            for batch in ld:
                cell = batch['cell_tokens'].to(device)
                motion = batch['motion_tokens'].to(device)
                time = batch['time_tokens'].to(device)
                labels = batch['label']
                out = encoder(cell, motion, time)
                for i in range(len(labels)):
                    if labels[i].item() == 0:
                        source_z_normals.append(out['z_norm'][i].cpu())

    source_z_normals = torch.stack(source_z_normals)  # (N, d)

    # Score target by distance to k-th nearest source normal
    test_loader = DataLoader(target_test_ds, batch_size=256, shuffle=False,
                             collate_fn=pad_batch)
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            cell = batch['cell_tokens'].to(device)
            motion = batch['motion_tokens'].to(device)
            time = batch['time_tokens'].to(device)
            out = encoder(cell, motion, time)
            z = out['z_norm'].cpu()
            sims = z @ source_z_normals.T  # (B, N)
            knn_sim, _ = sims.topk(10, dim=1)
            score = 1.0 - knn_sim.mean(dim=1).numpy()  # higher = more anomalous
            all_scores.extend(score.tolist())
            all_labels.extend(batch['label'].numpy().tolist())

    return eval_scores(all_labels, all_scores)


# ─────────────────────────────────────────────
# Baseline 3: Deep SAD
# ─────────────────────────────────────────────

def run_deep_sad(source_datasets, target_test_ds, k_shot_samples, args, device):
    """Deep Semi-supervised Anomaly Detection."""
    d = args.d_model
    encoder = TrajectoryEncoder(d_model=d, n_heads=4, n_layers=4,
                                dropout=0.1, n_domains=3).to(device)
    # Compute center from source normals
    combined = ConcatDataset(source_datasets)
    loader = DataLoader(combined, batch_size=128, shuffle=True,
                        collate_fn=pad_batch, drop_last=True)
    opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)

    # Init center
    encoder.eval()
    zs = []
    with torch.no_grad():
        for batch in loader:
            cell = batch['cell_tokens'].to(device)
            motion = batch['motion_tokens'].to(device)
            time = batch['time_tokens'].to(device)
            out = encoder(cell, motion, time)
            zs.append(out['z'])
            if len(zs) > 10:
                break
    center = torch.cat(zs).mean(0).detach()  # (d,)

    # Train: minimize dist to center for normals, maximize for anomalies
    encoder.train()
    for epoch in range(5):
        for batch in loader:
            cell = batch['cell_tokens'].to(device)
            motion = batch['motion_tokens'].to(device)
            time = batch['time_tokens'].to(device)
            labels = batch['label'].to(device)
            out = encoder(cell, motion, time)
            dist = ((out['z'] - center) ** 2).sum(-1)  # (B,)
            # normals: minimize dist; anomalies: maximize dist
            normal_loss = dist[labels == 0].mean() if (labels == 0).any() else 0
            anom_loss = (1.0 / (dist[labels == 1] + 1e-6)).mean() if (labels == 1).any() else 0
            loss = normal_loss + anom_loss
            opt.zero_grad(); loss.backward(); opt.step()

    # K-shot fine-tuning on target
    if k_shot_samples:
        for epoch in range(5):
            random.shuffle(k_shot_samples)
            batch = pad_batch(k_shot_samples[:32])
            cell = batch['cell_tokens'].to(device)
            motion = batch['motion_tokens'].to(device)
            time = batch['time_tokens'].to(device)
            labels = batch['label'].to(device)
            out = encoder(cell, motion, time)
            dist = ((out['z'] - center) ** 2).sum(-1)
            normal_loss = dist[labels == 0].mean() if (labels == 0).any() else 0
            anom_loss = (1.0 / (dist[labels == 1] + 1e-6)).mean() if (labels == 1).any() else 0
            loss = normal_loss + anom_loss
            opt.zero_grad(); loss.backward(); opt.step()

    # Evaluate
    encoder.eval()
    all_scores, all_labels = [], []
    test_loader = DataLoader(target_test_ds, batch_size=256, shuffle=False,
                             collate_fn=pad_batch)
    with torch.no_grad():
        for batch in test_loader:
            cell = batch['cell_tokens'].to(device)
            motion = batch['motion_tokens'].to(device)
            time = batch['time_tokens'].to(device)
            out = encoder(cell, motion, time)
            dist = ((out['z'] - center) ** 2).sum(-1).cpu().numpy()
            all_scores.extend(dist.tolist())
            all_labels.extend(batch['label'].numpy().tolist())
    return eval_scores(all_labels, all_scores)


# ─────────────────────────────────────────────
# Baseline 4: DANN + Transformer
# Domain-Adversarial Neural Network with same backbone
# ─────────────────────────────────────────────

def run_dann(source_datasets, target_train_ds, target_test_ds, args, device):
    """DANN with same Transformer backbone."""
    d = args.d_model
    encoder = TrajectoryEncoder(d_model=d, n_heads=4, n_layers=4,
                                dropout=0.1, n_domains=3).to(device)
    cls_head = nn.Linear(d, 1).to(device)

    # Combine source (labeled) + target (unlabeled for domain head)
    source_combined = ConcatDataset(source_datasets)
    source_loader = DataLoader(source_combined, batch_size=64, shuffle=True,
                               collate_fn=pad_batch, drop_last=True)
    target_loader = DataLoader(target_train_ds, batch_size=64, shuffle=True,
                               collate_fn=pad_batch, drop_last=True)

    params = list(encoder.parameters()) + list(cls_head.parameters())
    opt = torch.optim.Adam(params, lr=5e-4)

    encoder.train()
    cls_head.train()
    for epoch in range(5):
        target_iter = iter(target_loader)
        for batch_s in source_loader:
            try:
                batch_t = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                batch_t = next(target_iter)

            # Source classification loss
            cell_s = batch_s['cell_tokens'].to(device)
            motion_s = batch_s['motion_tokens'].to(device)
            time_s = batch_s['time_tokens'].to(device)
            y_s = batch_s['label'].float().to(device)
            domain_s = batch_s['domain_id'].to(device)

            out_s = encoder(cell_s, motion_s, time_s)
            cls_loss = F.binary_cross_entropy_with_logits(
                cls_head(out_s['z']).squeeze(-1), y_s)

            # Domain adversarial (source + target)
            cell_t = batch_t['cell_tokens'].to(device)
            motion_t = batch_t['motion_tokens'].to(device)
            time_t = batch_t['time_tokens'].to(device)
            domain_t = batch_t['domain_id'].to(device)

            out_t = encoder(cell_t, motion_t, time_t)
            dom_loss = (F.cross_entropy(out_s['domain_logits'], domain_s) +
                        F.cross_entropy(out_t['domain_logits'], domain_t))

            loss = cls_loss + 0.1 * dom_loss
            opt.zero_grad(); loss.backward(); opt.step()

    # Evaluate
    encoder.eval()
    cls_head.eval()
    all_scores, all_labels = [], []
    test_loader = DataLoader(target_test_ds, batch_size=256, shuffle=False,
                             collate_fn=pad_batch)
    with torch.no_grad():
        for batch in test_loader:
            cell = batch['cell_tokens'].to(device)
            motion = batch['motion_tokens'].to(device)
            time = batch['time_tokens'].to(device)
            out = encoder(cell, motion, time)
            prob = torch.sigmoid(cls_head(out['z']).squeeze(-1)).cpu().numpy()
            all_scores.extend(prob.tolist())
            all_labels.extend(batch['label'].numpy().tolist())
    return eval_scores(all_labels, all_scores)


# ─────────────────────────────────────────────
# Baseline 5: Prototypical Networks
# ─────────────────────────────────────────────

def run_protonet(source_datasets, target_test_ds, k_shot_samples, args, device):
    """Prototypical Networks with trajectory encoder."""
    d = args.d_model
    encoder = TrajectoryEncoder(d_model=d, n_heads=4, n_layers=4,
                                dropout=0.1, n_domains=3).to(device)

    # Episodic training on source
    combined = ConcatDataset(source_datasets)
    loader = DataLoader(combined, batch_size=128, shuffle=True,
                        collate_fn=pad_batch, drop_last=True)
    opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    encoder.train()

    for epoch in range(5):
        for batch in loader:
            cell = batch['cell_tokens'].to(device)
            motion = batch['motion_tokens'].to(device)
            time = batch['time_tokens'].to(device)
            labels = batch['label'].to(device)
            out = encoder(cell, motion, time)
            z = out['z_norm']

            # Compute prototypes
            proto_normal = z[labels == 0].mean(0) if (labels == 0).any() else z.mean(0)
            proto_anomaly = z[labels == 1].mean(0) if (labels == 1).any() else z.mean(0)
            prototypes = torch.stack([proto_normal, proto_anomaly])  # (2, d)

            # Squared Euclidean distance to prototypes → negative distance as logits
            dists = ((z.unsqueeze(1) - prototypes.unsqueeze(0)) ** 2).sum(-1)  # (B, 2)
            loss = F.cross_entropy(-dists, labels)
            opt.zero_grad(); loss.backward(); opt.step()

    # Use K-shot target labels to update prototypes
    encoder.eval()
    with torch.no_grad():
        if k_shot_samples:
            batch = pad_batch(k_shot_samples)
            cell = batch['cell_tokens'].to(device)
            motion = batch['motion_tokens'].to(device)
            time = batch['time_tokens'].to(device)
            labels = batch['label'].to(device)
            out = encoder(cell, motion, time)
            z = out['z_norm']
            proto_normal = z[labels == 0].mean(0) if (labels == 0).any() else proto_normal
            proto_anomaly = z[labels == 1].mean(0) if (labels == 1).any() else proto_anomaly
            prototypes = torch.stack([proto_normal, proto_anomaly])
        else:
            # Use source prototypes
            zs_normal, zs_anomaly = [], []
            for ds in source_datasets:
                ld = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=pad_batch)
                for b in ld:
                    cell = b['cell_tokens'].to(device)
                    motion = b['motion_tokens'].to(device)
                    time = b['time_tokens'].to(device)
                    out = encoder(cell, motion, time)
                    for i in range(len(b['label'])):
                        if b['label'][i] == 0:
                            zs_normal.append(out['z_norm'][i])
                        else:
                            zs_anomaly.append(out['z_norm'][i])
                    if len(zs_normal) > 5000:
                        break
            proto_normal = torch.stack(zs_normal).mean(0)
            proto_anomaly = torch.stack(zs_anomaly).mean(0) if zs_anomaly else proto_normal
            prototypes = torch.stack([proto_normal, proto_anomaly])

    # Evaluate
    all_scores, all_labels = [], []
    test_loader = DataLoader(target_test_ds, batch_size=256, shuffle=False,
                             collate_fn=pad_batch)
    with torch.no_grad():
        for batch in test_loader:
            cell = batch['cell_tokens'].to(device)
            motion = batch['motion_tokens'].to(device)
            time = batch['time_tokens'].to(device)
            out = encoder(cell, motion, time)
            z = out['z_norm']
            # Score = dist to normal - dist to anomaly (higher = more anomalous)
            dist_normal = ((z - prototypes[0]) ** 2).sum(-1)
            dist_anomaly = ((z - prototypes[1]) ** 2).sum(-1)
            score = (dist_normal - dist_anomaly).cpu().numpy()
            all_scores.extend(score.tolist())
            all_labels.extend(batch['label'].numpy().tolist())
    return eval_scores(all_labels, all_scores)


# ─────────────────────────────────────────────
# Baseline 6: Source-only Transformer (no transfer)
# ─────────────────────────────────────────────

def run_source_only(source_datasets, target_test_ds, args, device):
    """Train classifier on source, evaluate on target — no adaptation."""
    d = args.d_model
    encoder = TrajectoryEncoder(d_model=d, n_heads=4, n_layers=4,
                                dropout=0.1, n_domains=3).to(device)
    cls_head = nn.Linear(d, 1).to(device)

    combined = ConcatDataset(source_datasets)
    loader = DataLoader(combined, batch_size=128, shuffle=True,
                        collate_fn=pad_batch, drop_last=True)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(cls_head.parameters()), lr=1e-3)

    encoder.train(); cls_head.train()
    for epoch in range(5):
        for batch in loader:
            cell = batch['cell_tokens'].to(device)
            motion = batch['motion_tokens'].to(device)
            time = batch['time_tokens'].to(device)
            y = batch['label'].float().to(device)
            out = encoder(cell, motion, time)
            loss = F.binary_cross_entropy_with_logits(cls_head(out['z']).squeeze(-1), y)
            opt.zero_grad(); loss.backward(); opt.step()

    encoder.eval(); cls_head.eval()
    all_scores, all_labels = [], []
    test_loader = DataLoader(target_test_ds, batch_size=256, shuffle=False,
                             collate_fn=pad_batch)
    with torch.no_grad():
        for batch in test_loader:
            cell = batch['cell_tokens'].to(device)
            motion = batch['motion_tokens'].to(device)
            time = batch['time_tokens'].to(device)
            out = encoder(cell, motion, time)
            prob = torch.sigmoid(cls_head(out['z']).squeeze(-1)).cpu().numpy()
            all_scores.extend(prob.tolist())
            all_labels.extend(batch['label'].numpy().tolist())
    return eval_scores(all_labels, all_scores)


# ─────────────────────────────────────────────
# Baseline 7: Target-only oracle (upper bound)
# ─────────────────────────────────────────────

def run_target_oracle(target_train_ds, target_test_ds, args, device):
    """Full supervision on target — upper bound."""
    d = args.d_model
    encoder = TrajectoryEncoder(d_model=d, n_heads=4, n_layers=4,
                                dropout=0.1, n_domains=3).to(device)
    cls_head = nn.Linear(d, 1).to(device)

    loader = DataLoader(target_train_ds, batch_size=128, shuffle=True,
                        collate_fn=pad_batch, drop_last=True)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(cls_head.parameters()), lr=1e-3)

    encoder.train(); cls_head.train()
    for epoch in range(10):
        for batch in loader:
            cell = batch['cell_tokens'].to(device)
            motion = batch['motion_tokens'].to(device)
            time = batch['time_tokens'].to(device)
            y = batch['label'].float().to(device)
            out = encoder(cell, motion, time)
            loss = F.binary_cross_entropy_with_logits(cls_head(out['z']).squeeze(-1), y)
            opt.zero_grad(); loss.backward(); opt.step()

    encoder.eval(); cls_head.eval()
    all_scores, all_labels = [], []
    test_loader = DataLoader(target_test_ds, batch_size=256, shuffle=False,
                             collate_fn=pad_batch)
    with torch.no_grad():
        for batch in test_loader:
            cell = batch['cell_tokens'].to(device)
            motion = batch['motion_tokens'].to(device)
            time = batch['time_tokens'].to(device)
            out = encoder(cell, motion, time)
            prob = torch.sigmoid(cls_head(out['z']).squeeze(-1)).cpu().numpy()
            all_scores.extend(prob.tolist())
            all_labels.extend(batch['label'].numpy().tolist())
    return eval_scores(all_labels, all_scores)


# ─────────────────────────────────────────────
# Baseline 8: AdapTime + kNN (RATFM proxy)
# Majority-vote retrieval scoring
# ─────────────────────────────────────────────

def run_adaptime_knn(source_datasets, target_test_ds, args, device):
    """RATFM-style proxy: retrieve top-k from source memory, majority vote labels."""
    d = args.d_model
    encoder = TrajectoryEncoder(d_model=d, n_heads=4, n_layers=2,
                                dropout=0.1, n_domains=3).to(device)
    # Pretrain encoder
    combined = ConcatDataset(source_datasets)
    loader = DataLoader(combined, batch_size=128, shuffle=True,
                        collate_fn=pad_batch, drop_last=True)
    opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    encoder.train()
    for epoch in range(3):
        for batch in loader:
            cell = batch['cell_tokens'].to(device)
            motion = batch['motion_tokens'].to(device)
            time = batch['time_tokens'].to(device)
            mask = torch.rand_like(cell.float()) < 0.15
            out = encoder(cell, motion, time, mask=mask)
            loss = (F.cross_entropy(out['cell_logits'][mask], cell[mask]) +
                    F.cross_entropy(out['motion_logits'][mask], motion[mask]))
            opt.zero_grad(); loss.backward(); opt.step()

    # Build source memory with labels
    encoder.eval()
    source_z_all, source_labels_all = [], []
    with torch.no_grad():
        for ds in source_datasets:
            ld = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=pad_batch)
            for batch in ld:
                cell = batch['cell_tokens'].to(device)
                motion = batch['motion_tokens'].to(device)
                time = batch['time_tokens'].to(device)
                out = encoder(cell, motion, time)
                source_z_all.append(out['z_norm'].cpu())
                source_labels_all.append(batch['label'])
    source_z_all = torch.cat(source_z_all)
    source_labels_all = torch.cat(source_labels_all)

    # Score by majority vote of top-k retrieved labels
    test_loader = DataLoader(target_test_ds, batch_size=256, shuffle=False,
                             collate_fn=pad_batch)
    all_scores, all_labels = [], []
    k = 20
    with torch.no_grad():
        for batch in test_loader:
            cell = batch['cell_tokens'].to(device)
            motion = batch['motion_tokens'].to(device)
            time = batch['time_tokens'].to(device)
            out = encoder(cell, motion, time)
            z = out['z_norm'].cpu()
            sims = z @ source_z_all.T  # (B, N)
            _, topk_idx = sims.topk(k, dim=1)
            retrieved_labels = source_labels_all[topk_idx].float()  # (B, k)
            score = retrieved_labels.mean(dim=1).numpy()  # fraction of anomalies
            all_scores.extend(score.tolist())
            all_labels.extend(batch['label'].numpy().tolist())
    return eval_scores(all_labels, all_scores)
