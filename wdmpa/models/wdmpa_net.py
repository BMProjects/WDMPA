"""WDMPA-Net: Main gaze estimation model.

This model combines:
- AWWD (Adaptive Wavelet Weighted Downsampling) for efficient feature extraction
- MPA (Multi-scale Parallel Attention) for spatial and channel attention
- Adaptive Star Block for element-wise feature interaction
"""

import torch
from torch import nn
from timm.layers import DropPath, trunc_normal_

from wdmpa.modules.awwd import AWWD
from wdmpa.modules.star_block import AdaptiveStarBlock


class WDMPANet(nn.Module):
    """WDMPA-Net for gaze estimation.

    Args:
        base_dim: Base channel dimension. Default: 24.
        depths: Number of blocks in each stage. Default: [2, 2, 8, 3].
        mlp_ratio: MLP expansion ratio. Default: 4.
        drop_path_rate: Stochastic depth rate. Default: 0.0.
        num_classes: Number of output classes (2 for pitch/yaw). Default: 2.

    Example:
        >>> model = WDMPANet()
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # (1, 2) for pitch and yaw
    """

    def __init__(
        self,
        base_dim: int = 24,
        depths: list[int] = [2, 2, 8, 3],
        mlp_ratio: int = 4,
        drop_path_rate: float = 0.0,
        num_classes: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = base_dim

        # Stem: initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.in_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU6(inplace=True),
        )

        # Build stages with stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0

        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = AWWD(
                self.in_channel,
                embed_dim,
                need=False,
                fusion_type="weighted",
                use_gate=False,
            )
            self.in_channel = embed_dim

            blocks = [
                AdaptiveStarBlock(self.in_channel, mlp_ratio, dpr[cur + i])
                for i in range(depths[i_layer])
            ]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))

        # Head
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Calculate feature dimension after stages
        # For 224x224 input with 4 stages of 2x downsampling: 224 / 2^5 = 7
        # Channels: 24 * 2^3 = 192, feature_dim = 192 * 7 * 7 = 9408
        self.fc = nn.Linear(9408, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights."""
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Gaze prediction of shape (B, 2) for pitch and yaw in degrees.
        """
        x = self.stem(x)

        for stage in self.stages:
            x = stage(x)

        # Flatten and predict
        bs = x.shape[0]
        x = x.view(bs, -1)
        x = self.fc(x)

        return x

    def forward_with_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with grid output (for compatibility with original training code).

        Returns tensor of shape (B, 1, 2, 1, 1) for backward compatibility.
        """
        out = self.forward(x)
        bs = out.shape[0]
        return out.view(bs, 1, 2, 1, 1).permute(0, 1, 3, 4, 2).contiguous()


# Alias for backward compatibility
Emodel_StarNet = WDMPANet


if __name__ == "__main__":
    model = WDMPANet()
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    # Parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params / 1e6:.2f}M)")
