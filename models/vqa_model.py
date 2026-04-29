import torch
import torch.nn as nn
from models.vision_encoder import ResNetEncoder
from models.text_encoder import TextEncoder
from models.fusion import EarlyFusion, LateFusion, CrossModalFusion


class VQAModel(nn.Module):
    """
    Two-leg VQA model: ResNet-50 (vision) + DistilBERT (text) + fusion head.

    Each forward pass scores K (question, choice) pairs and returns logits
    [B, K] with padding choices masked to -inf.
    """

    def __init__(self, config):
        super().__init__()
        self.config      = config
        self.fusion_type = config.fusion_type

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

    def forward(self, image, input_ids, attention_mask, has_image, choice_mask):
        B, K, L = input_ids.shape

        text_feat = self.text_encoder(
            input_ids.view(B * K, L),
            attention_mask.view(B * K, L),
        )

        if self.fusion_type == "cross_modal":
            spatial, _ = self.vision_encoder(image, return_spatial=True)
            logits = self.fusion(spatial, text_feat, has_image)
        else:
            visual_feat = self.vision_encoder(image)
            logits      = self.fusion(visual_feat, text_feat, has_image)

        return logits.masked_fill(choice_mask == 0, float("-inf"))