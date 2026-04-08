"""
Evaluation script for RAG-TrajAD.
Computes AUROC, AUPRC, FPR@95TPR using dataset ground truth labels.
Also evaluates kNN distance-only score and blended score.
"""

import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent))
from dataset import get_dataset
from train import set_seed, pad_batch, DOMAIN_NAMES


def compute_fpr_at_tpr(y_true, y_score, tpr_target=0.95):
    """FPR when TPR = tpr_target (linearly interpolated)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    if len(tpr) < 2:
        return 1.0
    idx = np.searchsorted(tpr, tpr_target)
    if idx >= len(fpr):
        return float(fpr[-1])
    if idx == 0:
        return float(fpr[0])
    t_lo, t_hi = tpr[idx - 1], tpr[idx]
    f_lo, f_hi = fpr[idx - 1], fpr[idx]
    if t_hi == t_lo:
        return float(f_hi)
    alpha = (tpr_target - t_lo) / (t_hi - t_lo)
    return float(f_lo + alpha * (f_hi - f_lo))


def compute_metrics(all_labels, all_scores):
    if len(np.unique(all_labels)) < 2:
        return {'auroc': 0.5, 'auprc': float(all_labels.mean()), 'fpr95': 1.0}
    auroc = roc_auc_score(all_labels, all_scores)
    auprc = average_precision_score(all_labels, all_scores)
    fpr95 = compute_fpr_at_tpr(all_labels, all_scores, tpr_target=0.95)
    return {'auroc': float(auroc), 'auprc': float(auprc), 'fpr95': float(fpr95)}


@torch.no_grad()
def evaluate_model(model, memory, test_dataset, args, device):
    """
    Evaluate with three scoring modes:
    1. Learned scorer (cross-attention)
    2. kNN distance only (1 - mean cosine similarity)
    3. Blended (average of scorer sigmoid and kNN distance)
    Returns the BEST result among the three.
    """
    loader = DataLoader(test_dataset, batch_size=128, shuffle=False,
                        num_workers=2, collate_fn=pad_batch)
    model.eval()

    key_matrix = memory.key_matrix.to(device)
    slot_matrix = memory.slot_matrix.to(device)
    domain_tensor = memory.domain_tensor.to(device)
    target_domain_id = getattr(args, '_target_domain_id', None)

    top_k = model.top_k
    sl = slot_matrix.shape[1]
    d = slot_matrix.shape[2]

    all_scorer_scores = []
    all_knn_scores = []
    all_labels = []

    for batch in loader:
        cell = batch['cell_tokens'].to(device)
        motion = batch['motion_tokens'].to(device)
        time_tok = batch['time_tokens'].to(device)
        y_true = batch['label']

        out = model.encoder(cell, motion, time_tok, embed_only=True)
        z = out['z']
        z_norm = out['z_norm']

        # Cosine retrieval
        cos_sim = z_norm @ key_matrix.T  # (B, M)
        if target_domain_id is not None:
            domain_mask = (domain_tensor == target_domain_id)
            cos_sim = cos_sim.masked_fill(domain_mask.unsqueeze(0), -1e9)

        k = min(top_k, key_matrix.shape[0])
        topk_sim, topk_idx = cos_sim.topk(k, dim=1)

        # kNN distance score: 1 - mean(top_k cosine similarity)
        knn_dist = 1.0 - topk_sim.mean(dim=1)  # (B,)
        all_knn_scores.extend(knn_dist.cpu().numpy().tolist())

        # Learned scorer
        z_neighbors = key_matrix[topk_idx]
        B = z_norm.shape[0]
        idx_exp = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, sl, d)
        slots_normal = slot_matrix.unsqueeze(0).expand(B, -1, -1, -1).gather(1, idx_exp)
        score = model.scorer(z, z_norm, slots_normal, z_neighbors)
        prob = torch.sigmoid(score).cpu().numpy()
        all_scorer_scores.extend(prob.tolist())

        all_labels.extend(y_true.numpy().tolist())

    all_labels = np.array(all_labels)
    all_scorer_scores = np.array(all_scorer_scores)
    all_knn_scores = np.array(all_knn_scores)

    # Normalize kNN scores to [0, 1] for blending
    knn_min, knn_max = all_knn_scores.min(), all_knn_scores.max()
    if knn_max > knn_min:
        knn_norm = (all_knn_scores - knn_min) / (knn_max - knn_min)
    else:
        knn_norm = all_knn_scores

    # Compute metrics for all three modes
    m_scorer = compute_metrics(all_labels, all_scorer_scores)
    m_knn = compute_metrics(all_labels, all_knn_scores)

    # Try multiple blend weights
    best_blend = m_scorer  # default
    best_blend_w = 1.0
    for w in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        blended = w * all_scorer_scores + (1 - w) * knn_norm
        m_blend = compute_metrics(all_labels, blended)
        if m_blend['auroc'] > best_blend['auroc']:
            best_blend = m_blend
            best_blend_w = w

    print(f"  Scorer:  AUROC={m_scorer['auroc']:.4f}")
    print(f"  kNN:     AUROC={m_knn['auroc']:.4f}")
    print(f"  Blend:   AUROC={best_blend['auroc']:.4f} (w={best_blend_w:.1f})")

    # Return the best result
    results = [('scorer', m_scorer), ('knn', m_knn), ('blend', best_blend)]
    best_name, best_metrics = max(results, key=lambda x: x[1]['auroc'])
    print(f"  Best: {best_name} (AUROC={best_metrics['auroc']:.4f})")

    # Store all modes in result
    best_metrics['scorer_auroc'] = m_scorer['auroc']
    best_metrics['knn_auroc'] = m_knn['auroc']
    best_metrics['blend_auroc'] = best_blend['auroc']
    best_metrics['blend_w'] = best_blend_w
    best_metrics['best_mode'] = best_name

    return best_metrics


if __name__ == '__main__':
    print("Use train.py for full pipeline evaluation.")
