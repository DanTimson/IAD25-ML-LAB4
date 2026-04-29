import torch
import torch.nn as nn
from models.vision_encoder import ResNetEncoder
from models.text_encoder import TextEncoder
from models.fusion import EarlyFusion, LateFusion, CrossModalFusion


class VQAModel(nn.Module):
    """
    Two-leg VQA model:
      Leg 1 — ResNet-50 vision encoder  (non-transformer)
      Leg 2 — DistilBERT text encoder   (transformer; constraint is CV-only)
      Head  — one of {EarlyFusion, LateFusion, CrossModalFusion}

    Forward returns logits [B, max_choices]; padding choices are masked with -inf.
    """

    def __init__(self, config):
        super().__init__()
        self.config       = config
        self.fusion_type  = config.fusion_type
        self.max_choices  = config.max_choices

        self.vision_encoder = ResNetEncoder(config)
        self.text_encoder   = TextEncoder(config)

        if config.fusion_type == "early":
            self.fusion = EarlyFusion(config)
        elif config.fusion_type == "late":
            self.fusion = LateFusion(config)
        elif config.fusion_type == "cross_modal":
            self.fusion = CrossModalFusion(config)
        else:
            raise ValueError(f"Unknown fusion_type: {config.fusion_type!r}")

    def forward(
        self,
        image:          torch.Tensor,   # [B, 3, H, W]
        input_ids:      torch.Tensor,   # [B, max_choices, seq_len]
        attention_mask: torch.Tensor,   # [B, max_choices, seq_len]
        has_image:      torch.Tensor,   # [B]
        choice_mask:    torch.Tensor,   # [B, max_choices]  1=valid, 0=pad
    ) -> torch.Tensor:                  # [B, max_choices]
        B, K, L = input_ids.shape

        # ── text: encode all choices in one batched call ─────────────────
        ids_flat  = input_ids.view(B * K, L)
        mask_flat = attention_mask.view(B * K, L)
        text_feat = self.text_encoder(ids_flat, mask_flat)   # [B*K, D]

        # ── vision ───────────────────────────────────────────────────────
        need_spatial = (self.fusion_type == "cross_modal")

        if need_spatial:
            spatial, visual_feat = self.vision_encoder(image, return_spatial=True)
            logits = self.fusion(spatial, visual_feat, text_feat, has_image)
        else:
            visual_feat = self.vision_encoder(image)          # [B, D]
            logits      = self.fusion(visual_feat, text_feat, has_image)

        # Mask padding choices with a large negative value so softmax / argmax
        # never selects them
        logits = logits.masked_fill(choice_mask == 0, float("-inf"))
        return logits   # [B, K]

    # ── convenience helpers ───────────────────────────────────────────────────

    def get_gradcam_targets(self):
        """Return (activations, gradients) tensors stored by layer4 hooks."""
        return (self.vision_encoder.activations,
                self.vision_encoder.gradients)

    def get_cross_modal_attention(self) -> torch.Tensor | None:
        """
        Returns cross-modal attention weights from the last forward pass.
        Shape: [B*K, n_heads, 1, 49] → reshape to [B*K, n_heads, 7, 7] for viz.
        None if fusion_type != 'cross_modal'.
        """
        if self.fusion_type != "cross_modal":
            return None
        return self.fusion.attention_weights
