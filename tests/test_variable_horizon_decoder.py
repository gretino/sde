import pytest
import torch
from ecg_forecast.config import ModelConfig
from ecg_forecast.models.emission_decoder import EmissionDecoder as WaveformDecoder


def test_variable_horizon_decoder():
    cfg = ModelConfig()
    decoder = WaveformDecoder(
        latent_dim=cfg.latent_dim,
        context_dim=cfg.context_dim,
        num_leads=cfg.num_leads,
    )

    b = 2
    c_summary = torch.randn(b, cfg.context_dim)

    horizons = [(50, 13), (100, 25), (200, 50)]
    for target_samples, latent_steps in horizons:
        latent_path = torch.randn(b, latent_steps, cfg.latent_dim)
        wf_pred, _ = decoder(latent_path, c_summary, target_len=target_samples)

        assert wf_pred.shape == (b, target_samples, cfg.num_leads), f"Expected shape {(b, target_samples, cfg.num_leads)}, got {wf_pred.shape}"
