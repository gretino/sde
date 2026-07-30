import os
import pytest
import torch
from ecg_forecast.config import Config
from ecg_forecast.models.latent_sde_forecaster import LatentSDEForecaster
from ecg_forecast.training.trainer import Trainer


def test_best_checkpoint_stage_transition(tmp_path):
    cfg = Config()
    cfg.training.checkpoint_dir = str(tmp_path)
    model = LatentSDEForecaster(cfg.model)
    trainer = Trainer(model=model, config=cfg)

    # Save dummy best Stage A checkpoint with known weights
    best_path = os.path.join(tmp_path, "posterior_warmup_best.pt")
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.234)
    torch.save({"model_state_dict": model.state_dict()}, best_path)

    # Mutate model weights
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.0)

    # Trigger reloading
    loaded = trainer._load_best_stage_a_checkpoint()
    assert loaded, "Failed to load best Stage A checkpoint!"

    for p in model.parameters():
        assert torch.allclose(p, torch.tensor(1.234)), "Model weights were not reloaded from posterior_warmup_best.pt!"
