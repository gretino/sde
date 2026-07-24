import os
from typing import Optional, Tuple
import torch
import torch.nn as nn

from ..config import Config
from ..models.latent_sde_forecaster import LatentSDEForecaster


def load_forecaster_checkpoint(
    checkpoint_path: str,
    config: Optional[Config] = None,
    device: str = "cpu",
) -> Tuple[LatentSDEForecaster, Config]:
    """Loads LatentSDEForecaster model and config from a checkpoint file.
    
    If checkpoint file does not exist, raises FileNotFoundError immediately.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Required checkpoint file not found at '{checkpoint_path}'. "
            f"Cannot load forecaster model."
        )

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if config is None:
        if "config" in checkpoint and checkpoint["config"] is not None:
            config = checkpoint["config"]
        else:
            config = Config()

    model = LatentSDEForecaster(config=config.model)

    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    # Strip 'module.' prefix if saved from DataParallel
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, config
