"""Model variants for ablation study.

Provides variants of WDMPANet with different components removed/replaced
to support the ablation experiments in the paper.
"""

import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_

from wdmpa.modules.awwd import AWWD, HaarWavelet
from wdmpa.modules.mpa import MPA
from wdmpa.modules.star_block import AdaptiveStarBlock, Conv


# ============================================================================
# Ablation 1: AWWD variants
# ============================================================================

class StrideConvDownsample(nn.Module):
    """Standard stride-2 convolution downsampling (替换 AWWD)."""

    def __init__(self, dim_in: int, dim_out: int, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AWWDFixed(nn.Module):
    """AWWD with fixed (non-learnable) Haar wavelet weights."""

    def __init__(self, dim_in: int, dim_out: int, **kwargs):
        super().__init__()
        self.wavelet = HaarWavelet(dim_in, learnable=False)  # 关键：fixed
        self.dim = dim_in

        # 固定权重
        self.weight_h = nn.Parameter(torch.ones(1), requires_grad=False)
        self.weight_v = nn.Parameter(torch.ones(1), requires_grad=False)
        self.weight_d = nn.Parameter(torch.ones(1), requires_grad=False)

        self.conv_after_wave = nn.Sequential(
            nn.Conv2d(self.dim * 2, dim_out, kernel_size=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        haar = self.wavelet(x, inverse=False)
        a = haar.narrow(1, 0, self.dim)
        h = haar.narrow(1, self.dim, self.dim)
        v = haar.narrow(1, self.dim * 2, self.dim)
        d = haar.narrow(1, self.dim * 3, self.dim)

        high = h * self.weight_h + v * self.weight_v + d * self.weight_d
        x = torch.cat([a, high], dim=1)
        return self.conv_after_wave(x)


# ============================================================================
# Ablation 2: MPA variants
# ============================================================================

class ChannelAttentionOnly(nn.Module):
    """Only channel attention (SE-style), no spatial attention."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.global_avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        return x * self.sigmoid(y)


class SpatialAttentionOnly(nn.Module):
    """Only spatial attention with multi-scale dilated convolutions."""

    def __init__(self, in_channels: int, **kwargs):
        super().__init__()
        self.conv_3x3 = nn.Conv2d(in_channels, 1, 3, padding=1, dilation=1, bias=False)
        self.conv_5x5 = nn.Conv2d(in_channels, 1, 3, padding=2, dilation=2, bias=False)
        self.conv_7x7 = nn.Conv2d(in_channels, 1, 3, padding=3, dilation=3, bias=False)
        self.conv_fuse = nn.Conv2d(3, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([
            self.conv_3x3(x),
            self.conv_5x5(x),
            self.conv_7x7(x),
        ], dim=1)
        return x * self.sigmoid(self.conv_fuse(out))


class MPASingleScale(nn.Module):
    """MPA with single dilation rate (d=1 only)."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        # Channel attention
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()

        # Single scale spatial (only d=1)
        self.conv_spatial = nn.Conv2d(in_channels, 1, 3, padding=1, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel
        y = self.global_avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        x_channel = x * self.sigmoid_channel(y)

        # Spatial (single scale)
        spatial = self.conv_spatial(x_channel)
        return x_channel * self.sigmoid_spatial(spatial)


class NoAttention(nn.Module):
    """Identity module (no attention)."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ============================================================================
# Ablation Star Block variants
# ============================================================================

class AblationStarBlock(nn.Module):
    """Star Block with configurable attention module for ablation."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: int = 3,
        drop_path: float = 0.0,
        attention_type: str = "mpa",  # "mpa", "channel", "spatial", "single", "none"
        reduction: int = 16,
    ):
        super().__init__()
        self.dwconv = Conv(dim, dim, k=7, g=dim, act=False)

        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, kernel_size=1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, kernel_size=1)
        self.g = Conv(mlp_ratio * dim, dim, k=1, act=False)
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Configurable attention
        if attention_type == "mpa":
            self.attention = MPA(dim, reduction)
        elif attention_type == "channel":
            self.attention = ChannelAttentionOnly(dim, reduction)
        elif attention_type == "spatial":
            self.attention = SpatialAttentionOnly(dim)
        elif attention_type == "single":
            self.attention = MPASingleScale(dim, reduction)
        else:
            self.attention = NoAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = self.attention(x)
        return identity + self.drop_path(x)


# ============================================================================
# Full ablation model
# ============================================================================

class WDMPANetAblation(nn.Module):
    """WDMPANet variant for ablation study.

    Args:
        downsample_type: "awwd" (default), "awwd_fixed", "stride_conv"
        attention_type: "mpa" (default), "channel", "spatial", "single", "none"
    """

    def __init__(
        self,
        base_dim: int = 24,
        depths: list[int] = [2, 2, 8, 3],
        mlp_ratio: int = 4,
        drop_path_rate: float = 0.0,
        num_classes: int = 2,
        downsample_type: str = "awwd",
        attention_type: str = "mpa",
    ):
        super().__init__()
        self.downsample_type = downsample_type
        self.attention_type = attention_type
        self.in_channel = base_dim

        self.stem = nn.Sequential(
            nn.Conv2d(3, self.in_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU6(inplace=True),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0

        # Select downsampler
        if downsample_type == "awwd":
            DownsampleClass = AWWD
        elif downsample_type == "awwd_fixed":
            DownsampleClass = AWWDFixed
        else:
            DownsampleClass = StrideConvDownsample

        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = DownsampleClass(self.in_channel, embed_dim)
            self.in_channel = embed_dim

            blocks = [
                AblationStarBlock(
                    self.in_channel, mlp_ratio, dpr[cur + i],
                    attention_type=attention_type
                )
                for i in range(depths[i_layer])
            ]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))

        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(9408, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Convenience functions
def create_ablation_variants():
    """Create all ablation variants for export/testing."""
    variants = {
        # Full model
        "wdmpa_full": WDMPANetAblation(downsample_type="awwd", attention_type="mpa"),

        # AWWD ablations
        "wdmpa_awwd_fixed": WDMPANetAblation(downsample_type="awwd_fixed", attention_type="mpa"),
        "wdmpa_stride_conv": WDMPANetAblation(downsample_type="stride_conv", attention_type="mpa"),

        # MPA ablations
        "wdmpa_channel_only": WDMPANetAblation(downsample_type="awwd", attention_type="channel"),
        "wdmpa_spatial_only": WDMPANetAblation(downsample_type="awwd", attention_type="spatial"),
        "wdmpa_single_scale": WDMPANetAblation(downsample_type="awwd", attention_type="single"),
        "wdmpa_no_attention": WDMPANetAblation(downsample_type="awwd", attention_type="none"),
    }
    return variants


if __name__ == "__main__":
    # Test all variants
    print("Testing ablation variants...")
    variants = create_ablation_variants()

    x = torch.randn(1, 3, 224, 224)
    for name, model in variants.items():
        model.eval()
        with torch.no_grad():
            out = model(x)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  {name}: output={out.shape}, params={params:.2f}M")
