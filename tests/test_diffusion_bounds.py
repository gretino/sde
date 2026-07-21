import pytest
import torch
from ecg_forecast.config import ModelConfig
from ecg_forecast.models import LatentSDEForecaster


def test_diffusion_bounds_fixed_and_sigmoid():
    cfg = ModelConfig(num_leads=1, latent_dim=16, context_dim=64)
    model = LatentSDEForecaster(config=cfg)

    # Stage A & B: Fixed diffusion 0.01
    model.set_stage("A")
    assert torch.allclose(model.sde.sde_func.sigma, torch.full((16,), 0.01)), "Stage A diffusion must be fixed at 0.01"

    model.set_stage("B")
    assert torch.allclose(model.sde.sde_func.sigma, torch.full((16,), 0.01)), "Stage B diffusion must be fixed at 0.01"

    # Stage C: Bounded learnable [0.005, 0.050]
    model.set_stage("C")
    sigma_c = model.sde.sde_func.sigma
    assert (sigma_c >= 0.005).all() and (sigma_c <= 0.050).all(), f"Stage C diffusion out of bounds [0.005, 0.050]: {sigma_c}"
