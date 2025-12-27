"""Gaze estimation metrics.

The primary metric is angular error in degrees between predicted and ground truth gaze directions.
"""

import torch
import numpy as np


def gaze_to_3d(gaze: torch.Tensor) -> torch.Tensor:
    """Convert gaze angles (pitch, yaw) to 3D gaze direction vector.

    Args:
        gaze: Tensor of shape (..., 2) with (pitch, yaw) in degrees.

    Returns:
        3D gaze vector of shape (..., 3).
    
    Note:
        Uses MPIIGaze standard coordinate system:
        - pitch: vertical angle (up/down)
        - yaw: horizontal angle (left/right)
    """
    # Convert to radians
    pitch = gaze[..., 0] * torch.pi / 180
    yaw = gaze[..., 1] * torch.pi / 180

    # Convert to 3D direction (MPIIGaze standard)
    # x: horizontal (left-right), y: vertical (up-down), z: depth
    x = -torch.cos(pitch) * torch.sin(yaw)
    y = -torch.sin(pitch)
    z = -torch.cos(pitch) * torch.cos(yaw)

    return torch.stack([x, y, z], dim=-1)


def angular_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute angular error between predicted and target gaze.

    Args:
        pred: Predicted gaze of shape (B, 2) with (pitch, yaw) in degrees.
        target: Target gaze of shape (B, 2) with (pitch, yaw) in degrees.

    Returns:
        Angular error in degrees of shape (B,).

    Example:
        >>> pred = torch.tensor([[10.0, 20.0]])
        >>> target = torch.tensor([[12.0, 18.0]])
        >>> error = angular_error(pred, target)  # ~2.8 degrees
    """
    pred_3d = gaze_to_3d(pred)
    target_3d = gaze_to_3d(target)

    # Normalize
    pred_norm = torch.nn.functional.normalize(pred_3d, dim=-1)
    target_norm = torch.nn.functional.normalize(target_3d, dim=-1)

    # Compute angle
    cos_sim = (pred_norm * target_norm).sum(dim=-1)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

    error_rad = torch.acos(cos_sim)
    error_deg = error_rad * 180 / torch.pi

    return error_deg


def mean_angular_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute mean angular error.

    Args:
        pred: Predicted gaze of shape (N, 2).
        target: Target gaze of shape (N, 2).

    Returns:
        Mean angular error in degrees.
    """
    errors = angular_error(pred, target)
    return errors.mean().item()


if __name__ == "__main__":
    # Test metrics
    pred = torch.tensor([[10.0, 20.0], [5.0, -10.0]])
    target = torch.tensor([[12.0, 18.0], [6.0, -8.0]])

    errors = angular_error(pred, target)
    print(f"Angular errors: {errors}")
    print(f"Mean error: {mean_angular_error(pred, target):.2f}°")
