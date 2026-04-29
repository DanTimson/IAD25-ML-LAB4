import math
import torch
import torch.nn as nn


class EarlyFusion(nn.Module):
    """Concat [visual, text] -> MLP -> scalar score per choice."""

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

    def forward(self, visual_feat, text_feat, has_image, **kwargs):
        B = visual_feat.shape[0]
        K = text_feat.shape[0] // B

        vis  = visual_feat.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
        gate = has_image.unsqueeze(1).expand(-1, K).reshape(B * K, 1)
        vis  = vis * gate

        scores = self.mlp(torch.cat([vis, text_feat], dim=-1)).squeeze(-1)
        return scores.view(B, K)


class LateFusion(nn.Module):
    """
    Separate unimodal scoring heads combined via a learned gate.
    Visual contribution is suppressed when no image is present.
    """

    def __init__(self, config):
        super().__init__()
        D = config.fusion_dim
        self.visual_head = nn.Sequential(
            nn.Linear(D, D // 2), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(D // 2, 1),
        )
        self.text_head = nn.Sequential(
            nn.Linear(D, D // 2), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(D // 2, 1),
        )
        self.log_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, visual_feat, text_feat, has_image, **kwargs):
        B = visual_feat.shape[0]
        K = text_feat.shape[0] // B

        v_score    = self.visual_head(visual_feat).expand(-1, K).reshape(B * K, 1)
        t_score    = self.text_head(text_feat)
        alpha      = torch.sigmoid(self.log_alpha)
        image_gate = has_image.unsqueeze(1).expand(-1, K).reshape(B * K, 1)
        eff_alpha  = alpha * image_gate

        scores = eff_alpha * v_score + (1.0 - eff_alpha) * t_score
        return scores.squeeze(-1).view(B, K)


class CrossModalFusion(nn.Module):
    """
    Question-guided spatial attention over ResNet layer4 feature map.

    Query = text CLS token; Keys/Values = 49 spatial tokens from layer4.
    Attention weights [B*K, n_heads, 7, 7] are stored for visualisation.
    """

    def __init__(self, config):
        super().__init__()
        D = config.fusion_dim
        H = config.n_attn_heads
        assert D % H == 0

        self.n_heads  = H
        self.head_dim = D // H
        self.scale    = math.sqrt(self.head_dim)

        self.spatial_proj = nn.Sequential(nn.Linear(2048, D), nn.LayerNorm(D))
        self.q_proj   = nn.Linear(D, D)
        self.k_proj   = nn.Linear(D, D)
        self.v_proj   = nn.Linear(D, D)
        self.out_proj = nn.Linear(D, D)

        self.score_head = nn.Sequential(
            nn.Linear(D * 2, D), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(D, 1),
        )

        self.attention_weights: torch.Tensor | None = None

    def forward(self, spatial_feat, text_feat, has_image, **kwargs):
        B  = spatial_feat.shape[0]
        K  = text_feat.shape[0] // B
        BK = B * K
        D  = self.q_proj.out_features

        spatial     = self.spatial_proj(spatial_feat.flatten(2).permute(0, 2, 1))
        spatial_exp = spatial.unsqueeze(1).expand(-1, K, -1, -1).reshape(BK, 49, D)

        Q  = self.q_proj(text_feat).unsqueeze(1)
        Ks = self.k_proj(spatial_exp)
        Vs = self.v_proj(spatial_exp)

        def split_heads(t, seq):
            return t.view(BK, seq, self.n_heads, self.head_dim).transpose(1, 2)

        Q, Ks, Vs = split_heads(Q, 1), split_heads(Ks, 49), split_heads(Vs, 49)

        attn_w = torch.softmax(
            torch.matmul(Q, Ks.transpose(-2, -1)) / self.scale, dim=-1
        )
        self.attention_weights = attn_w.detach()

        attended = torch.matmul(attn_w, Vs).transpose(1, 2).reshape(BK, D)
        attended = self.out_proj(attended)

        image_gate = has_image.unsqueeze(1).expand(-1, K).reshape(BK, 1)
        attended   = attended * image_gate

        scores = self.score_head(torch.cat([attended, text_feat], dim=-1)).squeeze(-1)
        return scores.view(B, K)