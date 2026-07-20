import pytest
import torch

from ecg_forecast.config import ModelConfig
from ecg_forecast.models import LatentSDEForecaster


def test_model_shapes_lead1():
    cfg = ModelConfig(num_leads=1, latent_dim=32, context_dim=128)
    model = LatentSDEForecaster(config=cfg)

    b = 4
    c_wf = torch.randn(b, 500, 1)
    f_wf = torch.randn(b, 200, 1)

    post_out = model.forward_posterior(c_wf, f_wf)
    assert post_out.waveform_mean.shape == (b, 200, 1)
    assert post_out.waveform_scale.shape == (1,)
    assert post_out.latent_path.shape == (b, 50, 32)
    assert post_out.initial_kl is not None
    assert post_out.path_kl is not None

    prior_out = model.forward_prior(c_wf, num_samples=1)
    assert prior_out.waveform_mean.shape == (b, 200, 1)
    assert prior_out.latent_path.shape == (b, 50, 32)


def test_model_shapes_12lead():
    cfg = ModelConfig(num_leads=12, latent_dim=32, context_dim=128)
    model = LatentSDEForecaster(config=cfg)

    b = 2
    c_wf = torch.randn(b, 500, 12)
    f_wf = torch.randn(b, 200, 12)

    post_out = model.forward_posterior(c_wf, f_wf)
    assert post_out.waveform_mean.shape == (b, 200, 12)
    assert post_out.waveform_scale.shape == (12,)
    assert post_out.latent_path.shape == (b, 50, 32)
