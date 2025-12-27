"""Loss functions for gaze estimation."""

import torch
import torch.nn as nn

from wdmpa.utils.metrics import angular_error


class GazeLoss(nn.Module):
    """Combined loss for gaze estimation.

    Combines L1 loss on gaze angles with optional angular error regularization.

    Args:
        l1_weight: Weight for L1 loss. Default: 1.0.
        angular_weight: Weight for angular error loss. Default: 0.0.

    Example:
        >>> criterion = GazeLoss(l1_weight=1.0, angular_weight=0.01)
        >>> loss = criterion(pred, target)
    """

    def __init__(self, l1_weight: float = 1.0, angular_weight: float = 0.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.angular_weight = angular_weight
        self.l1_loss = nn.L1Loss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute loss.

        Args:
            pred: Predicted gaze of shape (B, 2).
            target: Target gaze of shape (B, 2).

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        # L1 loss
        l1 = self.l1_loss(pred, target)

        # Angular error (for monitoring, optionally as loss)
        ang_error = angular_error(pred, target).mean()

        # Total loss
        total = self.l1_weight * l1
        if self.angular_weight > 0:
            total = total + self.angular_weight * ang_error

        loss_dict = {
            "l1": l1.detach(),
            "angular": ang_error.detach(),
        }

        return total, loss_dict


class MSEGazeLoss(nn.Module):
    """MSE loss for gaze estimation."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss = self.mse(pred, target)
        ang_error = angular_error(pred, target).mean()

        return loss, {"mse": loss.detach(), "angular": ang_error.detach()}
