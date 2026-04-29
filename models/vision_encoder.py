import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    """
    ResNet-50 backbone (non-transformer) with two output modes:

      global_only=True  → [B, fusion_dim]                  (early / late fusion)
      global_only=False → ([B, 2048, 7, 7], [B, fusion_dim]) (cross-modal fusion)

    layer4 activations and gradients are stored after every forward pass
    so GradCAM can be computed without re-running the model.
    """

    def __init__(self, config):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Decompose into named stages so we can hook layer4
        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1,
                                     backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4     # output: [B, 2048, 7, 7]

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj        = nn.Sequential(
            nn.Linear(2048, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
        )
        self.dropout = nn.Dropout(config.dropout)

        # Storage for GradCAM (populated by _register_hooks)
        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None
        self._register_hooks()

    # ------------------------------------------------------------------
    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self._activations = out          # saved each forward pass

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0]    # saved on backward

        self.layer4.register_forward_hook(fwd_hook)
        self.layer4.register_full_backward_hook(bwd_hook)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, return_spatial: bool = False):
        """
        Args:
            x              : [B, 3, H, W]
            return_spatial : if True, also return layer4 feature map

        Returns:
            global_only=True  → projected [B, fusion_dim]
            global_only=False → (spatial [B, 2048, 7, 7],
                                  projected [B, fusion_dim])
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)             # [B, 2048, 7, 7]; hook fires here

        pooled    = self.global_pool(x).flatten(1)   # [B, 2048]
        projected = self.proj(self.dropout(pooled))  # [B, fusion_dim]

        if return_spatial:
            return x, projected
        return projected

    # ------------------------------------------------------------------
    @property
    def activations(self) -> torch.Tensor | None:
        return self._activations

    @property
    def gradients(self) -> torch.Tensor | None:
        return self._gradients