import pytest
import torch
from ecg_forecast.config import ModelConfig
from ecg_forecast.models.posterior_encoder import PosteriorEncoder


def test_variable_horizon_posterior():
    cfg = ModelConfig()
    post_encoder = PosteriorEncoder(
        num_leads=cfg.num_leads,
        context_dim=cfg.context_dim,
        latent_dim=cfg.latent_dim,
    )

    b = 2
    c_wf = torch.randn(b, 500, cfg.num_leads)
    c_summary = torch.randn(b, cfg.context_dim)

    horizons_sec = [0.5, 1.0, 2.0]
    for h_sec in horizons_sec:
        f_samples = int(round(h_sec * 100))
        t_latent = int(round(h_sec * 25))

        f_wf = torch.randn(b, f_samples, cfg.num_leads)
        full_wf = torch.cat([c_wf, f_wf], dim=1)

        summary, rec_path, mean, logvar = post_encoder(
            full_wf, c_summary, future_samples=f_samples, sampling_rate=100, latent_rate=25
        )

        assert rec_path.shape == (b, t_latent, cfg.context_dim), f"Expected rec_path shape {(b, t_latent, cfg.context_dim)}, got {rec_path.shape}"

        assert mean.shape == (b, cfg.latent_dim)
        assert logvar.shape == (b, cfg.latent_dim)
