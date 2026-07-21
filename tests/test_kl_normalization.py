import pytest
import torch
from ecg_forecast.config import ModelConfig
from ecg_forecast.models import LatentSDEForecaster


def test_kl_normalization_latent_dim_invariance():
    # Latent dim 16 vs 64
    cfg16 = ModelConfig(num_leads=1, latent_dim=16, context_dim=64)
    cfg64 = ModelConfig(num_leads=1, latent_dim=64, context_dim=64)

    m16 = LatentSDEForecaster(config=cfg16)
    m64 = LatentSDEForecaster(config=cfg64)

    p_m16 = torch.zeros(2, 16)
    p_logv16 = torch.zeros(2, 16)
    q_m16 = torch.ones(2, 16) * 0.5
    q_logv16 = torch.zeros(2, 16)

    p_m64 = torch.zeros(2, 64)
    p_logv64 = torch.zeros(2, 64)
    q_m64 = torch.ones(2, 64) * 0.5
    q_logv64 = torch.zeros(2, 64)

    kl16 = m16._gaussian_kl(p_m16, p_logv16, q_m16, q_logv16)
    kl64 = m64._gaussian_kl(p_m64, p_logv64, q_m64, q_logv64)

    # Initial KL per dimension should be identical (~0.125) regardless of latent_dim dimension
    assert abs(kl16.item() - kl64.item()) < 1e-4, f"KL16 ({kl16.item()}) != KL64 ({kl64.item()})"
