"""ResNet50 for Gaze Estimation (L2CS-Net Style).

A standard baseline using ResNet50 backbone, similar to L2CS-Net architecture.
This is a heavier model for accuracy comparison.

Reference: L2CS-Net (https://github.com/Ahmednull/L2CS-Net)
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Gaze(nn.Module):
    """ResNet50 based gaze estimation model (L2CS-Net style).

    Args:
        pretrained: Use ImageNet pretrained weights. Default: True.
        num_classes: Output dimension (2 for pitch/yaw). Default: 2.

    Example:
        >>> model = ResNet50Gaze(pretrained=True)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # (1, 2)
    """

    def __init__(self, pretrained: bool = True, num_classes: int = 2):
        super().__init__()

        # Load backbone
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
            backbone = resnet50(weights=weights)
        else:
            backbone = resnet50(weights=None)

        # Extract features (remove fc)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.avgpool = backbone.avgpool

        # Gaze regression head
        # ResNet50 outputs 2048 channels
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
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
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = ResNet50Gaze(pretrained=True)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(f"Output: {out.shape}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params / 1e6:.2f}M)")
