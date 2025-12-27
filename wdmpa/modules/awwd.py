"""AWWD: Adaptive Wavelet Weighted Downsampling.

A learnable downsampling module based on Haar wavelet transform that preserves
both low-frequency (approximation) and high-frequency (detail) information.
"""

import torch
from torch import nn
from torch.nn import functional as F


class HaarWavelet(nn.Module):
    """Haar wavelet transform for 2D feature maps.

    Decomposes input into four subbands:
    - LL (approximation): low-frequency in both dimensions
    - LH (horizontal): vertical edges
    - HL (vertical): horizontal edges
    - HH (diagonal): diagonal details

    Args:
        in_channels: Number of input channels.
        learnable: If True, wavelet weights are learnable. Default: True.
    """

    def __init__(self, in_channels: int, learnable: bool = True):
        super().__init__()
        self.in_channels = in_channels

        # Haar wavelet filter bank
        # Shape: (4, 1, 2, 2) for LL, LH, HL, HH
        haar_weights = torch.ones(4, 1, 2, 2)

        # LH (horizontal detail): [[ 1, -1], [ 1, -1]]
        haar_weights[1, 0, 0, 1] = -1
        haar_weights[1, 0, 1, 1] = -1

        # HL (vertical detail): [[ 1,  1], [-1, -1]]
        haar_weights[2, 0, 1, 0] = -1
        haar_weights[2, 0, 1, 1] = -1

        # HH (diagonal detail): [[ 1, -1], [-1,  1]]
        haar_weights[3, 0, 1, 0] = -1
        haar_weights[3, 0, 0, 1] = -1

        # Repeat for all input channels
        haar_weights = torch.cat([haar_weights] * self.in_channels, dim=0)
        self.haar_weights = nn.Parameter(haar_weights)
        self.haar_weights.requires_grad = learnable

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Apply Haar wavelet transform.

        Args:
            x: Input tensor of shape (B, C, H, W).
            inverse: If True, apply inverse transform.

        Returns:
            Transformed tensor. Forward: (B, 4C, H/2, W/2), Inverse: (B, C, 2H, 2W).
        """
        if not inverse:
            # Forward transform
            out = F.conv2d(
                x, self.haar_weights, bias=None, stride=2, groups=self.in_channels
            ) / 4.0

            # Reshape to separate subbands
            B, _, H, W = out.shape
            out = out.reshape(B, self.in_channels, 4, H, W)
            out = out.transpose(1, 2)  # (B, 4, C, H, W)
            out = out.reshape(B, 4 * self.in_channels, H, W)
            return out
        else:
            # Inverse transform
            B, C4, H, W = x.shape
            out = x.reshape(B, 4, self.in_channels, H, W)
            out = out.transpose(1, 2)  # (B, C, 4, H, W)
            out = out.reshape(B, self.in_channels * 4, H, W)
            return F.conv_transpose2d(
                out, self.haar_weights, bias=None, stride=2, groups=self.in_channels
            )


class AWWD(nn.Module):
    """Adaptive Wavelet Weighted Downsampling.

    Combines low-frequency approximation with weighted high-frequency details
    to preserve important edge information during downsampling.

    Args:
        dim_in: Input channel dimension.
        dim_out: Output channel dimension.
        need: If True, apply initial 1x1 conv to change channels. Default: False.
        fusion_type: How to fuse high-frequency bands ('sum', 'weighted', 'conv').
        use_gate: If True, use gating mechanism for high-frequency. Default: False.

    Example:
        >>> awwd = AWWD(24, 48)
        >>> x = torch.randn(1, 24, 112, 112)
        >>> out = awwd(x)  # (1, 48, 56, 56)
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        need: bool = False,
        fusion_type: str = "weighted",
        use_gate: bool = False,
    ):
        super().__init__()
        self.need = need
        self.use_gate = use_gate
        self.fusion_type = fusion_type

        # Optional channel adjustment
        if need:
            self.first_conv = nn.Conv2d(dim_in, dim_out, kernel_size=1)
            self.wavelet = HaarWavelet(dim_out, learnable=True)
            self.dim = dim_out
        else:
            self.wavelet = HaarWavelet(dim_in, learnable=True)
            self.dim = dim_in

        # High-frequency fusion
        if fusion_type == "sum":
            self.high_fusion = lambda h, v, d: h + v + d
        elif fusion_type == "weighted":
            self.weight_h = nn.Parameter(torch.ones(1))
            self.weight_v = nn.Parameter(torch.ones(1))
            self.weight_d = nn.Parameter(torch.ones(1))
            self.high_fusion = (
                lambda h, v, d: h * self.weight_h + v * self.weight_v + d * self.weight_d
            )
        elif fusion_type == "conv":
            self.high_fusion = nn.Conv2d(self.dim * 3, self.dim, kernel_size=1)

        # Optional gating
        if use_gate:
            self.gate_conv = nn.Sequential(
                nn.Conv2d(self.dim, self.dim, kernel_size=1),
                nn.BatchNorm2d(self.dim),
                nn.Sigmoid(),
            )

        # Output projection
        self.conv_after_wave = nn.Sequential(
            nn.Conv2d(self.dim * 2, dim_out, kernel_size=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Output tensor of shape (B, C_out, H/2, W/2).
        """
        if self.need:
            x = self.first_conv(x)

        # Wavelet decomposition
        haar = self.wavelet(x, inverse=False)

        # Split into subbands
        a = haar.narrow(1, 0, self.dim)  # LL (approximation)
        h = haar.narrow(1, self.dim, self.dim)  # LH (horizontal)
        v = haar.narrow(1, self.dim * 2, self.dim)  # HL (vertical)
        d = haar.narrow(1, self.dim * 3, self.dim)  # HH (diagonal)

        # Fuse high-frequency bands
        if self.fusion_type == "conv":
            high = self.high_fusion(torch.cat([h, v, d], dim=1))
        else:
            high = self.high_fusion(h, v, d)

        # Apply gating if enabled
        if self.use_gate:
            gate = self.gate_conv(a)
            high = high * gate

        # Concatenate and project
        x = torch.cat([a, high], dim=1)
        x = self.conv_after_wave(x)

        return x


if __name__ == "__main__":
    # Test AWWD module
    dim_in, dim_out = 24, 48
    x = torch.randn(1, dim_in, 112, 112)
    awwd = AWWD(dim_in, dim_out, need=False, fusion_type="weighted", use_gate=False)
    out = awwd(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
