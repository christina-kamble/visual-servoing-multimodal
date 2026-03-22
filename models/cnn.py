"""
Convolutional Neural Network (CNN)
Processes 128×128 RGB images and outputs feature vectors
for use as a standalone model or within the Hybrid architecture.
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    A lightweight CNN baseline for mapping images to trajectory weights.

    Architecture:
        Conv(3→16) → ReLU → MaxPool
        Conv(16→32) → ReLU → MaxPool
        FC(32768→256) → ReLU
        FC(256→output_size)

    Args:
        output_size (int): Number of output weights (default: 10)
    """
    def __init__(self, output_size: int = 10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class CNNEncoder(nn.Module):
    """
    CNN encoder used as the visual branch within the Hybrid model.
    Outputs a 64-dimensional feature vector.

    Architecture:
        Conv(3→32) → Pool → Dropout(0.25)  [×3]
        Conv(32→64)
        FC(16384→64)

    Input:  (B, 3, 128, 128)
    Output: (B, 64)
    """
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3,  32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 16 * 16, 64), nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
