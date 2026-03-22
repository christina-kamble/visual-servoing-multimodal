"""
Custom PyTorch Dataset for multi-modal visual servoing data.

Handles loading of:
  - Encoded images (.png via PIL)
  - Target weights (.npy)
  - Joint trajectory arrays (.npy)
  - Task trajectory arrays (.npy)
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class VisualServoingDataset(Dataset):
    """
    Dataset for multi-modal visual servoing.

    Loads encoded images alongside trajectory and weight data.
    Aligns all modalities to the shortest available split to avoid index errors.

    Args:
        img_dir       (str): Path to directory of encoded images
        weights_dir   (str): Path to directory of .npy weight files
        traj_joint_dir(str): Path to directory of joint trajectory .npy files
        traj_task_dir (str): Path to directory of task trajectory .npy files
        transform     (optional): torchvision transform pipeline for images

    Returns per item:
        img        (Tensor): Transformed image tensor  (C, H, W)
        traj_data  (Tensor): Concatenated [joint; task] trajectory (flat)
        weights    (Tensor): Target weight vector
    """

    def __init__(
        self,
        img_dir: str,
        weights_dir: str,
        traj_joint_dir: str,
        traj_task_dir: str,
        transform: Optional[transforms.Compose] = None,
    ):
        self.img_dir = img_dir
        self.weights_dir = weights_dir
        self.traj_joint_dir = traj_joint_dir
        self.traj_task_dir = traj_task_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        img_files        = sorted(os.listdir(img_dir))
        weight_files     = sorted(os.listdir(weights_dir))
        traj_joint_files = sorted(os.listdir(traj_joint_dir))
        traj_task_files  = sorted(os.listdir(traj_task_dir))

        # Align all modalities to the smallest split
        n = min(len(img_files), len(weight_files), len(traj_joint_files), len(traj_task_files))
        self.img_files        = img_files[:n]
        self.weight_files     = weight_files[:n]
        self.traj_joint_files = traj_joint_files[:n]
        self.traj_task_files  = traj_task_files[:n]

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = Image.open(os.path.join(self.img_dir, self.img_files[idx])).convert("RGB")
        img = self.transform(img)

        weights = torch.tensor(
            np.load(os.path.join(self.weights_dir, self.weight_files[idx])),
            dtype=torch.float32
        )
        traj_joint = torch.tensor(
            np.load(os.path.join(self.traj_joint_dir, self.traj_joint_files[idx])),
            dtype=torch.float32
        )
        traj_task = torch.tensor(
            np.load(os.path.join(self.traj_task_dir, self.traj_task_files[idx])),
            dtype=torch.float32
        )

        traj_data = torch.cat((traj_joint.flatten(), traj_task.flatten()), dim=0)
        return img, traj_data, weights


def get_dataloader(
    img_dir: str,
    weights_dir: str,
    traj_joint_dir: str,
    traj_task_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Convenience function to build a DataLoader from raw data directories.

    Args:
        img_dir, weights_dir, traj_joint_dir, traj_task_dir: data paths
        batch_size  (int):  samples per batch
        shuffle     (bool): shuffle each epoch
        num_workers (int):  parallel data loading workers

    Returns:
        torch.utils.data.DataLoader
    """
    dataset = VisualServoingDataset(img_dir, weights_dir, traj_joint_dir, traj_task_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
