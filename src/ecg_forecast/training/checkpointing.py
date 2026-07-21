import os
from typing import Dict, Any, Optional
import torch
from ..data.preprocessing import PREPROCESSING_VERSION


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    stage: str,
    metrics: Dict[str, float],
    global_step: int = 0,
    scheduler: Optional[Any] = None,
    config: Optional[Any] = None,
    record_splits: Optional[Dict[str, Any]] = None,
):
    """Saves checkpoint dictionary matching Section 13 specification."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "stage": stage,
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
        "validation_metrics": metrics,
        "record_splits": record_splits,
        "preprocessing_version": PREPROCESSING_VERSION,
    }
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Loads checkpoint matching Section 13 specification into model, optimizer, and scheduler."""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint
