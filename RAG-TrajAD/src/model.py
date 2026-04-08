"""
RAG-TrajAD model: Encoder + Memory + Contrastive Retrieval + Cross-Attention Scorer
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from dataset import CELL_VOCAB_SIZE, MOTION_BINS, TIME_BINS


# ─────────────────────────────────────────────
# Module 1: Shared Transformer Encoder
# ─────────────────────────────────────────────

class TrajectoryEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        n_domains: int = 3,
        mask_prob: float = 0.15,
    ):
        super().__init__()
        self.d_model = d_model
        self.mask_prob = mask_prob

        # Embeddings
        self.cell_emb = nn.Embedding(CELL_VOCAB_SIZE + 1, d_model, padding_idx=0)
        self.motion_emb = nn.Embedding(MOTION_BINS + 1, d_model, padding_idx=0)
        self.time_emb = nn.Embedding(TIME_BINS + 1, d_model, padding_idx=0)

        # Position encoding
        self.pos_enc = nn.Embedding(256, d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Projection head (for contrastive / memory key)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Gradient-reversal domain classifier
        self.domain_classifier = nn.Sequential(
            GradientReversal(alpha=1.0),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_domains),
        )

        # Masked token reconstruction heads
        self.cell_head = nn.Linear(d_model, CELL_VOCAB_SIZE)
        self.motion_head = nn.Linear(d_model, MOTION_BINS)

    def forward(
        self,
        cell_tokens: torch.Tensor,    # (B, T)
        motion_tokens: torch.Tensor,  # (B, T)
        time_tokens: torch.Tensor,    # (B, T)
        mask: Optional[torch.Tensor] = None,  # (B, T) bool, True = masked
        embed_only: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, T = cell_tokens.shape
        pos = torch.arange(T, device=cell_tokens.device).unsqueeze(0).expand(B, -1)

        # Token embeddings
        x = (self.cell_emb(cell_tokens) +
             self.motion_emb(motion_tokens) +
             self.time_emb(time_tokens) +
             self.pos_enc(pos))

        # Apply mask for pretraining (replace with zero)
        if mask is not None:
            x = x * (~mask).float().unsqueeze(-1)

        x = self.transformer(x)   # (B, T, d)

        # CLS = mean pool
        z = x.mean(dim=1)         # (B, d)
        z_norm = F.normalize(self.proj(z), dim=-1)

        if embed_only:
            return {
                'z': z,
                'z_norm': z_norm,
                'token_features': x,
            }

        domain_logits = self.domain_classifier(z)

        # Reconstruction logits (for masked positions)
        cell_logits = self.cell_head(x)    # (B, T, V_cell)
        motion_logits = self.motion_head(x)  # (B, T, V_motion)

        return {
            'z': z,
            'z_norm': z_norm,
            'domain_logits': domain_logits,
            'cell_logits': cell_logits,
            'motion_logits': motion_logits,
            'token_features': x,
        }


class GradientReversal(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def set_alpha(self, alpha: float):
        self.alpha = alpha

    def forward(self, x):
        return _GradientReversalFunction.apply(x, self.alpha)


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


# ─────────────────────────────────────────────
# Module 2: Cross-Domain Memory
# ─────────────────────────────────────────────

class CrossDomainMemory:
    """
    Stores trajectory embeddings with labels, domain IDs, and subtype tags.
    All source data only — NO target data is ever added.
    """
    def __init__(self, d_model: int = 128, slot_len: int = 8):
        self.d_model = d_model
        self.slot_len = slot_len

        self.keys: List[torch.Tensor] = []       # (d,) normalized embeddings
        self.slots: List[torch.Tensor] = []      # (slot_len, d) compressed slots
        self.labels: List[int] = []
        self.domains: List[int] = []
        self.subtypes: List[int] = []            # cluster ID from k-means
        self.subtype_names: List[str] = []       # raw subtype string

    def add(self, z_norm: torch.Tensor, token_features: torch.Tensor,
            label: int, domain_id: int, subtype_name: str, subtype_id: int):
        self.keys.append(z_norm.detach().cpu())
        # Compress token features to slot_len via mean-pooling segments
        T, d = token_features.shape
        if T >= self.slot_len:
            seg_size = T // self.slot_len
            slots = torch.stack([
                token_features[i * seg_size:(i + 1) * seg_size].mean(0)
                for i in range(self.slot_len)
            ])
        else:
            pad = token_features.mean(0, keepdim=True).expand(self.slot_len - T, -1)
            slots = torch.cat([token_features, pad], dim=0)
        self.slots.append(slots.detach().cpu())
        self.labels.append(label)
        self.domains.append(domain_id)
        self.subtype_names.append(subtype_name)
        self.subtypes.append(subtype_id)

    def build_index(self):
        """Stack all keys into a matrix for fast retrieval."""
        self.key_matrix = torch.stack(self.keys)     # (N, d)
        self.slot_matrix = torch.stack(self.slots)    # (N, slot_len, d)
        self.label_tensor = torch.tensor(self.labels)
        self.domain_tensor = torch.tensor(self.domains)
        self.subtype_tensor = torch.tensor(self.subtypes)

    def retrieve_coarse(self, query_z: torch.Tensor, top_k: int = 200,
                        exclude_domain: Optional[int] = None) -> torch.Tensor:
        """Return top_k indices by cosine similarity."""
        device = query_z.device
        key_mat = self.key_matrix.to(device)
        dom_tensor = self.domain_tensor.to(device)
        sims = (query_z @ key_mat.T)  # (B, N)
        if exclude_domain is not None:
            mask = (dom_tensor == exclude_domain).unsqueeze(0)  # (1, N)
            sims = sims.masked_fill(mask, -1e9)
        _, indices = sims.topk(min(top_k, len(self.keys)), dim=-1)
        return indices   # (B, top_k)

    def __len__(self):
        return len(self.keys)


# ─────────────────────────────────────────────
# Module 3: Cross-Attention Anomaly Scorer (v7)
# Normal-only retrieval + cross-attention on token features
# ─────────────────────────────────────────────

class CrossAttentionScorer(nn.Module):
    """
    Cross-attention scorer operating on token-level features from retrieved normals.
    Query slots attend to retrieved normal trajectory tokens.
    Score = how different the query is from normal patterns.
    """
    def __init__(self, d_model: int = 128, n_query_slots: int = 8, n_heads: int = 4):
        super().__init__()
        self.query_slots = nn.Parameter(torch.randn(n_query_slots, d_model))

        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Encoder head for direct anomaly signal
        self.encoder_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # Final head: [pool(Q_after_attn), normal_dist(1), enc_logit(1)]
        in_dim = d_model + 2
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        z: torch.Tensor,                  # (B, d) raw encoder output
        z_norm: torch.Tensor,             # (B, d) normalized projection
        slots_normal: torch.Tensor,       # (B, N_norm, slot_len, d) retrieved normal slots
        z_normal_keys: torch.Tensor,      # (B, K, d) normal keys for distance
    ) -> torch.Tensor:
        B = z.shape[0]
        Q = self.query_slots.unsqueeze(0).expand(B, -1, -1)  # (B, n_slots, d)

        # Flatten retrieved normal slots: (B, N*slot_len, d)
        N_norm = slots_normal.shape[1]
        S_N = slots_normal.view(B, N_norm * slots_normal.shape[2], -1)

        # Cross-attention: query slots attend to normal token features
        E_N, _ = self.cross_attn(Q, S_N, S_N)
        E_N = self.norm1(E_N + Q)

        q_pool = E_N.mean(dim=1)  # (B, d)

        # Normal distance feature: 1 - mean cosine sim to retrieved normals
        cos_sim = F.cosine_similarity(
            z_norm.unsqueeze(1), z_normal_keys, dim=-1)  # (B, K)
        normal_dist = 1.0 - cos_sim.mean(dim=1, keepdim=True)  # (B, 1)

        # Encoder head logit
        enc_logit = self.encoder_head(z)  # (B, 1)

        feat = torch.cat([q_pool, normal_dist, enc_logit], dim=-1)
        score = self.head(feat).squeeze(-1)  # (B,)
        return score


# ─────────────────────────────────────────────
# Full RAG-TrajAD model
# ─────────────────────────────────────────────

class RAGTrajAD(nn.Module):
    """v7: Normal-only memory + CrossAttention on token slots + encoder head."""
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 4,
        n_query_slots: int = 8,
        n_domains: int = 3,
        top_k: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k

        self.encoder = TrajectoryEncoder(
            d_model=d_model, n_heads=n_heads, n_layers=n_encoder_layers,
            dropout=dropout, n_domains=n_domains,
        )
        self.scorer = CrossAttentionScorer(
            d_model=d_model, n_query_slots=n_query_slots, n_heads=n_heads)

    def score_with_memory(
        self,
        z: torch.Tensor,                # (B, d) raw encoder output
        z_norm: torch.Tensor,           # (B, d) normalized projection
        memory: CrossDomainMemory,
        exclude_domain: Optional[int] = None,
    ) -> torch.Tensor:
        """Cosine retrieval of top-K normal slots + cross-attention scoring."""
        device = z_norm.device
        key_matrix = memory.key_matrix.to(device)
        slot_matrix = memory.slot_matrix.to(device)

        cos_sim = z_norm @ key_matrix.T  # (B, M)

        if exclude_domain is not None:
            domain_tensor = memory.domain_tensor.to(device)
            domain_mask = (domain_tensor == exclude_domain)
            cos_sim = cos_sim.masked_fill(domain_mask.unsqueeze(0), -1e9)

        k = min(self.top_k, key_matrix.shape[0])
        _, topk_idx = cos_sim.topk(k, dim=1)  # (B, K)

        # Gather normal keys and slots
        z_neighbors = key_matrix[topk_idx]  # (B, K, d)
        # Gather slots: (B, K, slot_len, d)
        d = slot_matrix.shape[-1]
        sl = slot_matrix.shape[1]  # slot_len
        idx_exp = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, sl, d)
        slots_normal = slot_matrix.unsqueeze(0).expand(z_norm.shape[0], -1, -1, -1).gather(1, idx_exp)

        return self.scorer(z, z_norm, slots_normal, z_neighbors)  # (B,)
