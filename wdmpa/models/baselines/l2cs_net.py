"""L2CS-Net for Gaze Estimation.

This is a faithful reproduction of the official L2CS-Net implementation.
Reference: https://github.com/Ahmednull/L2CS-Net

L2CS-Net uses ResNet50 backbone with two separate heads for pitch and yaw,
treating gaze estimation as a classification problem with binned angles.
"""

import math
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """ResNet BasicBlock."""

    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet Bottleneck block."""

    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class L2CS(nn.Module):
    """L2CS-Net: Official implementation for gaze estimation.

    Uses ResNet backbone with two separate heads for pitch and yaw.
    Each head outputs num_bins logits for classification.

    Args:
        block: ResNet block type (BasicBlock or Bottleneck).
        layers: Number of blocks per stage.
        num_bins: Number of bins for angle classification. Default: 90.

    Example:
        >>> # ResNet50 version
        >>> model = L2CS(Bottleneck, [3, 4, 6, 3], num_bins=90)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> yaw, pitch = model(x)  # (1, 90), (1, 90)
    """

    def __init__(self, block, layers: list[int], num_bins: int = 90):
        super().__init__()
        self.inplanes = 64
        self.num_bins = num_bins

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Separate heads for yaw and pitch (official design)
        self.fc_yaw_gaze = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch_gaze = nn.Linear(512 * block.expansion, num_bins)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Gaze prediction (yaw, pitch)
        pre_yaw_gaze = self.fc_yaw_gaze(x)
        pre_pitch_gaze = self.fc_pitch_gaze(x)

        return pre_yaw_gaze, pre_pitch_gaze


def L2CS_ResNet18(num_bins: int = 90) -> L2CS:
    """L2CS-Net with ResNet18 backbone."""
    return L2CS(BasicBlock, [2, 2, 2, 2], num_bins=num_bins)


def L2CS_ResNet50(num_bins: int = 90) -> L2CS:
    """L2CS-Net with ResNet50 backbone (official)."""
    return L2CS(Bottleneck, [3, 4, 6, 3], num_bins=num_bins)


# Wrapper for our baseline format
class L2CSNet(nn.Module):
    """L2CS-Net wrapper for unified baseline interface.

    Wraps official L2CS implementation to match our baseline API.

    Args:
        num_bins: Number of bins for classification. Default: 90.
        backbone: "resnet50" or "resnet18". Default: "resnet50".
    """

    def __init__(self, num_bins: int = 90, backbone: str = "resnet50"):
        super().__init__()
        self.num_bins = num_bins

        if backbone == "resnet50":
            self.model = L2CS_ResNet50(num_bins)
        else:
            self.model = L2CS_ResNet18(num_bins)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (yaw_logits, pitch_logits)."""
        return self.model(x)

    def get_gaze_angles(self, x: torch.Tensor) -> torch.Tensor:
        """Get gaze angles (pitch, yaw) in radians using soft-argmax.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            Gaze angles (B, 2) as (pitch, yaw) in radians.
        """
        yaw_logits, pitch_logits = self(x)

        # Soft-argmax
        idx_tensor = torch.arange(self.num_bins, device=x.device, dtype=torch.float32)

        pitch_prob = torch.softmax(pitch_logits, dim=1)
        yaw_prob = torch.softmax(yaw_logits, dim=1)

        pitch_idx = torch.sum(pitch_prob * idx_tensor, dim=1)
        yaw_idx = torch.sum(yaw_prob * idx_tensor, dim=1)

        # Convert bin index to angle (bins cover -90 to +90 degrees)
        # idx_tensor ranges from 0 to num_bins-1
        # Angle = (idx - num_bins/2) * (180 / num_bins) degrees
        pitch = (pitch_idx - self.num_bins / 2) * (math.pi / self.num_bins)
        yaw = (yaw_idx - self.num_bins / 2) * (math.pi / self.num_bins)

        return torch.stack([pitch, yaw], dim=1)


class L2CSNetSimple(nn.Module):
    """Simplified L2CS-Net with direct regression for fair comparison.

    Uses same ResNet50 backbone but with direct angle regression
    instead of classification, matching other baseline APIs.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Build ResNet50 backbone
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Direct regression head
        self.fc = nn.Linear(2048, num_classes)

        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

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
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    print("Testing L2CS-Net (Official)...")
    model = L2CS_ResNet50(num_bins=90)
    x = torch.randn(1, 3, 224, 224)
    yaw, pitch = model(x)
    print(f"  Yaw: {yaw.shape}, Pitch: {pitch.shape}")

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {params:.2f}M")

    print("\nTesting L2CSNet wrapper...")
    wrapper = L2CSNet(num_bins=90)
    gaze = wrapper.get_gaze_angles(x)
    print(f"  Gaze angles: {gaze.shape}")

    print("\nTesting L2CSNetSimple...")
    simple = L2CSNetSimple()
    out = simple(x)
    print(f"  Output: {out.shape}")
    params = sum(p.numel() for p in simple.parameters()) / 1e6
    print(f"  Parameters: {params:.2f}M")
