import pytest
import torch
from ecg_forecast.config import ModelConfig
from ecg_forecast.models import LatentSDEForecaster


def test_observation_scale_bounds_fixed_and_sigmoid():
    cfg = ModelConfig(num_leads=12, latent_dim=32, context_dim=128)
    model = LatentSDEForecaster(config=cfg)

    # Stage A & B: Fixed scale 0.10
    model.set_stage("A")
    assert torch.allclose(model.decoder.observation_scale, torch.full((12,), 0.10)), "Stage A obs scale must be fixed at 0.10"

    model.set_stage("B")
    assert torch.allclose(model.decoder.observation_scale, torch.full((12,), 0.10)), "Stage B obs scale must be fixed at 0.10"

    # Stage C: Bounded learnable [0.03, 0.30]
    model.set_stage("C")
    scale_c = model.decoder.observation_scale
    assert (scale_c >= 0.03).all() and (scale_c <= 0.30).all(), f"Stage C scale out of bounds [0.03, 0.30]: {scale_c}"
