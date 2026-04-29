import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    """
    ResNet-50 backbone with GradCAM hooks on layer4.

    forward(x)                    -> [B, fusion_dim]
    forward(x, return_spatial=True) -> ([B, 2048, 7, 7], [B, fusion_dim])
    """

    def __init__(self, config):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(2048, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
        )
        self.dropout = nn.Dropout(config.dropout)

        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self):
        self.layer4.register_forward_hook(
            lambda m, i, o: setattr(self, "_activations", o)
        )
        self.layer4.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_gradients", go[0])
        )

    def forward(self, x: torch.Tensor, return_spatial: bool = False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        pooled    = self.global_pool(x).flatten(1)
        projected = self.proj(self.dropout(pooled))

        if return_spatial:
            return x, projected
        return projected

    @property
    def activations(self) -> torch.Tensor | None:
        return self._activations

    @property
    def gradients(self) -> torch.Tensor | None:
        return self._gradients