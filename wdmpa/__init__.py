"""WDMPA: Wavelet-based Downsampling with Multi-scale Parallel Attention for Gaze Estimation.

A lightweight gaze estimation model designed for edge deployment.

Example:
    >>> from wdmpa import WDMPANet
    >>> model = WDMPANet()
    >>> output = model(image_tensor)  # (B, 2) pitch and yaw
"""

__version__ = "0.1.0"
__author__ = "WDMPA Team"

from wdmpa.models.wdmpa_net import WDMPANet

__all__ = ["WDMPANet", "__version__"]
