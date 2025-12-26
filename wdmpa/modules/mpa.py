"""MPA: Multi-scale Parallel Attention.

A dual-branch attention module that applies:
1. Channel attention via squeeze-and-excitation
2. Spatial attention via multi-scale dilated convolutions
"""

import torch
import torch.nn as nn


class MPA(nn.Module):
    """Multi-scale Parallel Attention module.

    Applies channel attention followed by spatial attention with
    multi-scale receptive fields (3x3, 5x5, 7x7 via dilation).

    Args:
        in_channels: Number of input channels.
        reduction: Channel reduction ratio for SE block. Default: 16.

    Example:
        >>> mpa = MPA(64, reduction=16)
        >>> x = torch.randn(1, 64, 28, 28)
        >>> out = mpa(x)  # Same shape as input
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()

        # Channel attention (Squeeze-and-Excitation)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial attention with multi-scale dilated convolutions
        # All use 3x3 kernel but with different dilation rates for 3x3, 5x5, 7x7 receptive fields
        self.conv_3x3 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, dilation=1, bias=False)
        self.conv_5x5 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=2, dilation=2, bias=False)
        self.conv_7x7 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=3, dilation=3, bias=False)

        self.conv_fuse = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Attention-weighted tensor of same shape.
        """
        # Channel attention
        y = self.global_avg_pool(x)  # (B, C, 1, 1)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        channel_att = self.sigmoid_channel(y)  # (B, C, 1, 1)
        x_channel = x * channel_att

        # Spatial attention with multi-scale
        out_3x3 = self.conv_3x3(x_channel)
        out_5x5 = self.conv_5x5(x_channel)
        out_7x7 = self.conv_7x7(x_channel)

        # Fuse multi-scale features
        out_fused = torch.cat([out_3x3, out_5x5, out_7x7], dim=1)  # (B, 3, H, W)
        out_fused = self.conv_fuse(out_fused)  # (B, 1, H, W)
        spatial_att = self.sigmoid_spatial(out_fused)

        return x_channel * spatial_att


if __name__ == "__main__":
    # Test MPA module
    x = torch.randn(1, 64, 28, 28)
    mpa = MPA(64, reduction=16)
    out = mpa(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
