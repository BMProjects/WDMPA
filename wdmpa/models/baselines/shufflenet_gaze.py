"""ShuffleNetV2 for Gaze Estimation.

A lightweight baseline using ShuffleNetV2 backbone designed for edge devices.
"""

import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights


class ShuffleNetV2Gaze(nn.Module):
    """ShuffleNetV2 based gaze estimation model.

    Args:
        pretrained: Use ImageNet pretrained weights. Default: True.
        num_classes: Output dimension (2 for pitch/yaw). Default: 2.

    Example:
        >>> model = ShuffleNetV2Gaze(pretrained=True)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # (1, 2)
    """

    def __init__(self, pretrained: bool = True, num_classes: int = 2):
        super().__init__()

        # Load backbone
        if pretrained:
            weights = ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
            backbone = shufflenet_v2_x1_0(weights=weights)
        else:
            backbone = shufflenet_v2_x1_0(weights=None)

        # Extract features (remove fc)
        self.conv1 = backbone.conv1
        self.maxpool = backbone.maxpool
        self.stage2 = backbone.stage2
        self.stage3 = backbone.stage3
        self.stage4 = backbone.stage4
        self.conv5 = backbone.conv5

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Gaze regression head
        # ShuffleNetV2 x1.0 outputs 1024 channels
        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

        self._init_head()

    def _init_head(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = ShuffleNetV2Gaze(pretrained=True)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(f"Output: {out.shape}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params / 1e6:.2f}M)")
