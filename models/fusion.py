"""
Three fusion strategies for the two-leg VQA model.

All modules share the same interface:
  forward(...) → logits [B, max_choices]

The scoring approach is pairwise: for each sample, we compute one scalar
score per (visual_context, question+choice_i) pair, then pick argmax.
This handles variable numbers of choices naturally via masking upstream.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Early fusion
# ──────────────────────────────────────────────────────────────────────────────
class EarlyFusion(nn.Module):
    """
    Concatenate [visual_feat, text_feat] → MLP → scalar score per choice.

    Visual and textual representations are merged at the feature level before
    any classification decision is made.
    """

    def __init__(self, config):
        super().__init__()
        D = config.fusion_dim
        self.mlp = nn.Sequential(
            nn.Linear(D * 2, D),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, 1),
        )

    def forward(
        self,
        visual_feat: torch.Tensor,      # [B, D]
        text_feat:   torch.Tensor,      # [B * K, D]
        has_image:   torch.Tensor,      # [B]
        **kwargs,
    ) -> torch.Tensor:                  # [B, K]
        B = visual_feat.shape[0]
        K = text_feat.shape[0] // B

        # Expand visual to match every choice
        vis = visual_feat.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)

        # Zero-out visual contribution where image is absent
        gate = has_image.unsqueeze(1).expand(-1, K).reshape(B * K, 1)
        vis  = vis * gate

        fused  = torch.cat([vis, text_feat], dim=-1)     # [B*K, 2D]
        scores = self.mlp(fused).squeeze(-1)              # [B*K]
        return scores.view(B, K)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Late fusion
# ──────────────────────────────────────────────────────────────────────────────
class LateFusion(nn.Module):
    """
    Each modality scores independently; outputs are combined with a learnable gate.

    Visual branch: maps image feature → scalar (shared across choices).
    Text branch:   maps (question+choice) feature → scalar per choice.
    Gate: learned sigmoid weight; automatically suppressed when image is absent.
    """

    def __init__(self, config):
        super().__init__()
        D = config.fusion_dim
        self.visual_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(D // 2, 1),
        )
        self.text_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(D // 2, 1),
        )
        # Scalar gate: how much to trust vision vs. text
        self.log_alpha = nn.Parameter(torch.zeros(1))   # sigmoid(0) = 0.5

    def forward(
        self,
        visual_feat: torch.Tensor,      # [B, D]
        text_feat:   torch.Tensor,      # [B * K, D]
        has_image:   torch.Tensor,      # [B]
        **kwargs,
    ) -> torch.Tensor:                  # [B, K]
        B = visual_feat.shape[0]
        K = text_feat.shape[0] // B

        v_score = self.visual_head(visual_feat)          # [B, 1]
        v_score = v_score.expand(-1, K).reshape(B * K, 1)

        t_score = self.text_head(text_feat)              # [B*K, 1]

        alpha     = torch.sigmoid(self.log_alpha)        # scalar in (0, 1)
        image_gate = has_image.unsqueeze(1).expand(-1, K).reshape(B * K, 1)
        eff_alpha  = alpha * image_gate                  # 0 when no image

        scores = eff_alpha * v_score + (1.0 - eff_alpha) * t_score
        return scores.squeeze(-1).view(B, K)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Cross-modal attention fusion
# ──────────────────────────────────────────────────────────────────────────────
class CrossModalFusion(nn.Module):
    """
    Question-guided spatial attention over CNN feature map.

    Architecture:
      - Project ResNet layer4 spatial tokens [B, 49, 2048] → [B, 49, D]
      - Multi-head attention: Query = text CLS, Key/Value = spatial tokens
      - Attended visual context [B*K, D] is concatenated with text and scored

    The stored attention_weights [B*K, n_heads, 1, 49] can be reshaped to
    [B*K, n_heads, 7, 7] and overlaid on the original image for visualization.
    """

    def __init__(self, config):
        super().__init__()
        D = config.fusion_dim
        H = config.n_attn_heads
        assert D % H == 0, "fusion_dim must be divisible by n_attn_heads"

        self.n_heads  = H
        self.head_dim = D // H
        self.scale    = math.sqrt(self.head_dim)

        # Project raw CNN spatial features into fusion_dim
        self.spatial_proj = nn.Sequential(
            nn.Linear(2048, D),
            nn.LayerNorm(D),
        )

        self.q_proj   = nn.Linear(D, D)
        self.k_proj   = nn.Linear(D, D)
        self.v_proj   = nn.Linear(D, D)
        self.out_proj = nn.Linear(D, D)

        self.score_head = nn.Sequential(
            nn.Linear(D * 2, D),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(D, 1),
        )

        # Populated during forward; used by visualize/gradcam.py
        self.attention_weights: torch.Tensor | None = None

    def forward(
        self,
        spatial_feat:  torch.Tensor,    # [B, 2048, 7, 7]
        visual_global: torch.Tensor,    # [B, D]   (unused in scoring but kept for API consistency)
        text_feat:     torch.Tensor,    # [B * K, D]
        has_image:     torch.Tensor,    # [B]
        **kwargs,
    ) -> torch.Tensor:                  # [B, K]
        B  = visual_global.shape[0]
        K  = text_feat.shape[0] // B
        BK = B * K
        D  = visual_global.shape[-1]

        # ── spatial tokens ──────────────────────────────────────────────
        # [B, 2048, 7, 7] → [B, 49, 2048] → [B, 49, D]
        spatial = spatial_feat.flatten(2).permute(0, 2, 1)   # [B, 49, 2048]
        spatial = self.spatial_proj(spatial)                  # [B, 49, D]

        # Expand to match all choices: [B*K, 49, D]
        spatial_exp = (spatial
                       .unsqueeze(1)
                       .expand(-1, K, -1, -1)
                       .reshape(BK, 49, D))

        # ── multi-head cross-attention ───────────────────────────────────
        Q  = self.q_proj(text_feat).unsqueeze(1)     # [BK, 1, D]
        Ks = self.k_proj(spatial_exp)                 # [BK, 49, D]
        Vs = self.v_proj(spatial_exp)                 # [BK, 49, D]

        def split_heads(t, seq):
            return t.view(BK, seq, self.n_heads, self.head_dim).transpose(1, 2)

        Q  = split_heads(Q, 1)    # [BK, H, 1, hd]
        Ks = split_heads(Ks, 49)  # [BK, H, 49, hd]
        Vs = split_heads(Vs, 49)  # [BK, H, 49, hd]

        attn_logits = torch.matmul(Q, Ks.transpose(-2, -1)) / self.scale  # [BK, H, 1, 49]
        attn_w      = torch.softmax(attn_logits, dim=-1)

        # Detach to avoid retaining the graph; GradCAM uses hooks instead
        self.attention_weights = attn_w.detach()

        attended = torch.matmul(attn_w, Vs)                  # [BK, H, 1, hd]
        attended = attended.transpose(1, 2).reshape(BK, D)   # [BK, D]
        attended = self.out_proj(attended)                    # [BK, D]

        # Gate visual context by image presence
        image_gate = has_image.unsqueeze(1).expand(-1, K).reshape(BK, 1)
        attended   = attended * image_gate

        fused  = torch.cat([attended, text_feat], dim=-1)    # [BK, 2D]
        scores = self.score_head(fused).squeeze(-1)           # [BK]
        return scores.view(B, K)
