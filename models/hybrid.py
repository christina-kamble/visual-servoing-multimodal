"""
CNN-MLP Hybrid Model
Fuses visual features (from CNNEncoder) with trajectory features (from DeepMLP)
for multi-modal visual servoing weight prediction.
"""

import torch
import torch.nn as nn

from models.cnn import CNNEncoder
from models.mlp import DeepMLP


class HybridModel(nn.Module):
    """
    Multi-modal hybrid architecture combining a CNN visual branch
    and a deep MLP trajectory branch.

    Data flow:
        visual_input    (B, 3, 128, 128) → CNNEncoder  → (B, 64)
        trajectory_input (B, 49152)      → DeepMLP     → (B, 64)
                                                          ↓
                                            Concatenate → (B, 128)
                                            FC(128 → output_size)

    Args:
        output_size (int): Number of weights to predict (default: 10)
    """
    def __init__(self, output_size: int = 10):
        super(HybridModel, self).__init__()
        self.visual_branch = CNNEncoder()
        self.trajectory_branch = DeepMLP()
        self.head = nn.Linear(128, output_size)

    def forward(self, visual_input: torch.Tensor, trajectory_input: torch.Tensor) -> torch.Tensor:
        visual_features = self.visual_branch(visual_input)           # (B, 64)
        trajectory_features = self.trajectory_branch(trajectory_input)  # (B, 64)
        fused = torch.cat((visual_features, trajectory_features), dim=1)  # (B, 128)
        return self.head(fused)
