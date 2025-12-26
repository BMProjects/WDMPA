"""MobileNetV3-Small for Gaze Estimation.

A lightweight baseline using MobileNetV3-Small backbone with a gaze regression head.
Suitable for edge deployment comparison.
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MobileNetV3Gaze(nn.Module):
    """MobileNetV3-Small based gaze estimation model.

    Args:
        pretrained: Use ImageNet pretrained weights. Default: True.
        num_classes: Output dimension (2 for pitch/yaw). Default: 2.

    Example:
        >>> model = MobileNetV3Gaze(pretrained=True)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # (1, 2)
    """

    def __init__(self, pretrained: bool = True, num_classes: int = 2):
        super().__init__()

        # Load backbone
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            backbone = mobilenet_v3_small(weights=weights)
        else:
            backbone = mobilenet_v3_small(weights=None)

        # Remove classifier
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # Gaze regression head
        # MobileNetV3-Small outputs 576 channels
        self.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

        self._init_head()

    def _init_head(self):
        """Initialize the gaze head."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = MobileNetV3Gaze(pretrained=True)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(f"Output: {out.shape}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params / 1e6:.2f}M)")
