"""Star Block: Element-wise multiplication network block.

Based on the StarNet architecture, using element-wise multiplication
instead of addition for feature interaction.
"""

import torch
import torch.nn as nn
from timm.layers import DropPath

from wdmpa.modules.mpa import MPA


def autopad(k: int, p: int | None = None, d: int = 1) -> int:
    """Calculate padding for 'same' convolution."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with BatchNorm and activation."""

    default_act = nn.SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act if act is True
            else act if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class StarBlock(nn.Module):
    """Basic Star Block with element-wise multiplication.

    Uses the formula: output = input + drop_path(dwconv2(g(act(f1(x)) * f2(x))))

    Args:
        dim: Input/output channel dimension.
        mlp_ratio: Expansion ratio for hidden dimension. Default: 4.
        drop_path: Drop path rate. Default: 0.0.
    """

    def __init__(self, dim: int, mlp_ratio: int = 4, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = Conv(dim, dim, k=7, g=dim, act=False)
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, kernel_size=1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, kernel_size=1)
        self.g = Conv(mlp_ratio * dim, dim, k=1, act=False)
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2  # Element-wise multiplication
        x = self.dwconv2(self.g(x))
        return identity + self.drop_path(x)


class AdaptiveStarBlock(nn.Module):
    """Adaptive Star Block with Multi-scale Parallel Attention.

    Extends StarBlock with MPA attention after the multiplication stage.

    Args:
        dim: Input/output channel dimension.
        mlp_ratio: Expansion ratio for hidden dimension. Default: 3.
        drop_path: Drop path rate. Default: 0.0.
        reduction: Channel reduction for MPA. Default: 16.
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: int = 3,
        drop_path: float = 0.0,
        reduction: int = 16,
    ):
        super().__init__()
        self.dwconv = Conv(dim, dim, k=7, g=dim, act=False)
        self.mpa = MPA(in_channels=dim, reduction=reduction)

        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, kernel_size=1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, kernel_size=1)
        self.g = Conv(mlp_ratio * dim, dim, k=1, act=False)
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dwconv(x)

        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2  # Element-wise multiplication
        x = self.dwconv2(self.g(x))
        x = self.mpa(x)  # Apply attention

        return identity + self.drop_path(x)


if __name__ == "__main__":
    # Test blocks
    x = torch.randn(1, 64, 28, 28)

    star = StarBlock(64, mlp_ratio=4, drop_path=0.1)
    out1 = star(x)
    print(f"StarBlock: {x.shape} -> {out1.shape}")

    adaptive = AdaptiveStarBlock(64, mlp_ratio=3, drop_path=0.1)
    out2 = adaptive(x)
    print(f"AdaptiveStarBlock: {x.shape} -> {out2.shape}")
