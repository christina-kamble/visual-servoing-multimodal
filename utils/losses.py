"""
Loss functions for visual servoing weight prediction.
"""

import torch
import torch.nn as nn


def euclidean_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Euclidean (L2) loss averaged over the batch.

    Computes the mean Euclidean distance between predicted and target weight vectors.

    Args:
        predictions (Tensor): Model output  (B, D)
        targets     (Tensor): Ground truth  (B, D)

    Returns:
        Scalar loss tensor
    """
    return torch.sqrt(torch.sum((predictions - targets) ** 2, dim=1)).mean()


class EuclideanLoss(nn.Module):
    """nn.Module wrapper around euclidean_loss for use in training pipelines."""

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return euclidean_loss(predictions, targets)
