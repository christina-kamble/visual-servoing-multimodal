"""
Fully Connected Multi-Layer Perceptron (MLP)
Used as a trajectory encoder within the Hybrid model,
and also as a standalone baseline.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    A simple 3-layer MLP for mapping flattened image input to trajectory weights.
    Used as a standalone baseline model.

    Args:
        input_size (int): Size of flattened input (default: 128*128*3 = 49152)
        hidden_size (int): Number of neurons in hidden layers (default: 256)
        output_size (int): Number of output weights to predict (default: 10)
    """
    def __init__(self, input_size: int = 128 * 128 * 3, hidden_size: int = 256, output_size: int = 10):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DeepMLP(nn.Module):
    """
    A deep 13-layer MLP used as the trajectory branch within the Hybrid model.
    Progressively reduces dimensionality from 49152 → 64.

    Input: flattened trajectory data (128*128*3)
    Output: 64-dimensional feature vector
    """
    def __init__(self):
        super(DeepMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(128 * 128 * 3, 2816), nn.ReLU(),
            nn.Linear(2816, 2560), nn.ReLU(),
            nn.Linear(2560, 2304), nn.ReLU(),
            nn.Linear(2304, 2048), nn.ReLU(),
            nn.Linear(2048, 1792), nn.ReLU(),
            nn.Linear(1792, 1536), nn.ReLU(),
            nn.Linear(1536, 1280), nn.ReLU(),
            nn.Linear(1280, 1024), nn.ReLU(),
            nn.Linear(1024, 768),  nn.ReLU(),
            nn.Linear(768, 512),   nn.ReLU(),
            nn.Linear(512, 256),   nn.ReLU(),
            nn.Linear(256, 128),   nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
