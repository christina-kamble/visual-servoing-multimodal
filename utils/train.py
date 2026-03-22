"""
Training loop and evaluation utilities for visual servoing models.
"""

from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: Callable,
    device: torch.device,
    dual_input: bool = False,
) -> float:
    """
    Run one full training epoch.

    Args:
        model      : PyTorch model
        loader     : DataLoader yielding (visuals, trajectories, targets)
        optimizer  : Optimiser instance
        criterion  : Loss function
        device     : torch.device
        dual_input : True for HybridModel (takes both visual + trajectory inputs)

    Returns:
        Average loss over the epoch
    """
    model.train()
    total_loss = 0.0

    for visuals, trajectories, targets in loader:
        visuals, trajectories, targets = (
            visuals.to(device), trajectories.to(device), targets.to(device)
        )
        optimizer.zero_grad()

        if dual_input:
            outputs = model(visuals, trajectories)
        else:
            outputs = model(visuals)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def train(
    model: nn.Module,
    loader: DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-3,
    criterion: Optional[Callable] = None,
    device: Optional[torch.device] = None,
    dual_input: bool = False,
) -> List[float]:
    """
    Full training loop with epoch-level loss logging.

    Args:
        model      : Model to train
        loader     : DataLoader
        num_epochs : Number of training epochs
        lr         : Learning rate for Adam
        criterion  : Loss function (defaults to MSELoss)
        device     : Computation device (defaults to CUDA if available)
        dual_input : True for HybridModel

    Returns:
        List of per-epoch average losses
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = criterion or nn.MSELoss()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, loader, optimizer, criterion, device, dual_input)
        losses.append(avg_loss)
        print(f"Epoch [{epoch}/{num_epochs}]  Loss: {avg_loss:.4f}")

    print("Training complete.")
    return losses


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
    dual_input: bool = False,
    threshold: float = 0.1,
) -> Dict[str, float]:
    """
    Evaluate model and return regression metrics.

    Metrics returned:
        - MSE  : Mean Squared Error
        - MAE  : Mean Absolute Error
        - R2   : R² coefficient of determination
        - ACC  : Proportion of predictions within `threshold` of true value

    Args:
        model      : Trained model
        loader     : DataLoader
        device     : Computation device
        dual_input : True for HybridModel
        threshold  : Relative tolerance for accuracy metric (default 10%)

    Returns:
        Dict of metric name → float value
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    all_preds, all_targets = [], []

    for visuals, trajectories, targets in loader:
        visuals, trajectories = visuals.to(device), trajectories.to(device)

        if dual_input:
            preds = model(visuals, trajectories)
        else:
            preds = model(visuals)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.numpy())

    preds_np   = np.concatenate(all_preds,   axis=0)
    targets_np = np.concatenate(all_targets, axis=0)

    mse = mean_squared_error(targets_np, preds_np)
    mae = mean_absolute_error(targets_np, preds_np)
    r2  = r2_score(targets_np, preds_np)

    # Accuracy: proportion of predictions within `threshold` of true value
    relative_error = np.abs(preds_np - targets_np) / (np.abs(targets_np) + 1e-8)
    acc = float(np.mean(relative_error <= threshold))

    metrics = {"MSE": mse, "MAE": mae, "R2": r2, "Accuracy": acc}

    print("\n── Evaluation Results ──────────────────")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print("─" * 40)

    return metrics
