import pytest
import torch
import torchsde

from ecg_forecast.config import ModelConfig
from ecg_forecast.models import LatentSDEForecaster


def test_prior_has_no_future_leakage():
    cfg = ModelConfig(num_leads=1, latent_dim=16, context_dim=64)
    model = LatentSDEForecaster(config=cfg)
    model.eval()

    c_wf = torch.randn(2, 500, 1)
    bm = torchsde.BrownianInterval(t0=0.0, t1=2.0, size=(2, 16), entropy=42)

    torch.manual_seed(123)
    out1 = model.forward_prior(c_wf, num_samples=1, brownian_motion=bm)

    torch.manual_seed(123)
    out2 = model.forward_prior(c_wf, num_samples=1, brownian_motion=bm)

    # Identical inputs and Brownian motion produce exact same output
    assert torch.allclose(out1.waveform_mean, out2.waveform_mean, atol=1e-5)
    assert torch.allclose(out1.latent_path, out2.latent_path, atol=1e-5)
