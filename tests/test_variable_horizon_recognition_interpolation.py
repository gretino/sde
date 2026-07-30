import pytest
import torch
from ecg_forecast.config import ModelConfig
from ecg_forecast.models.conditional_sde import ConditionalLatentSDE as ConditionalSDE
from ecg_forecast.utils.timegrid import make_latent_times


def test_variable_horizon_recognition_interpolation():
    cfg = ModelConfig()
    sde = ConditionalSDE(latent_dim=cfg.latent_dim, context_dim=cfg.context_dim)

    b = 2
    for h_sec in [0.5, 1.0, 2.0]:
        f_samples = int(round(h_sec * 100))
        t_latent = int(round(h_sec * 25))

        rec_path = torch.randn(b, t_latent, cfg.latent_dim)
        ts = make_latent_times(future_samples=f_samples, sampling_rate=100, latent_rate=25, device="cpu")

        interp = sde._interpolate_recognition(rec_path, ts)
        assert interp.shape == (b, len(ts), cfg.latent_dim), f"Expected shape {(b, len(ts), cfg.latent_dim)}, got {interp.shape}"
