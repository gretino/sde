import pytest
import torch
from ecg_forecast.config import ModelConfig
from ecg_forecast.models.conditional_sde import ConditionalLatentSDE as ConditionalSDE
from ecg_forecast.utils.timegrid import make_latent_times


def test_deterministic_sde_mode():
    cfg = ModelConfig()
    sde = ConditionalSDE(latent_dim=cfg.latent_dim, context_dim=cfg.context_dim)
    
    b = 4
    z0 = torch.randn(b, cfg.latent_dim)
    c_summary = torch.randn(b, cfg.context_dim)
    ts = make_latent_times(future_samples=50, sampling_rate=100, latent_rate=25, device="cpu")

    # 1. Deterministic run 1 vs run 2
    torch.manual_seed(42)
    path1, _ = sde.integrate(z0, ts, context_summary=c_summary, mode="prior", deterministic=True)
    
    torch.manual_seed(1234)
    path2, _ = sde.integrate(z0, ts, context_summary=c_summary, mode="prior", deterministic=True)

    assert torch.allclose(path1, path2, atol=1e-6), "Deterministic integration produced different results on different seeds!"

    # 2. Stochastic run
    torch.manual_seed(42)
    path_stoch, _ = sde.integrate(z0, ts, context_summary=c_summary, mode="prior", deterministic=False)
    
    assert not torch.allclose(path1, path_stoch, atol=1e-4), "Deterministic and stochastic paths were unexpectedly identical!"
