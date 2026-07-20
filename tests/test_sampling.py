import pytest
import torch
import torchsde

from ecg_forecast.config import ModelConfig
from ecg_forecast.models import LatentSDEForecaster


def test_multi_sample_prior_generation():
    cfg = ModelConfig(num_leads=1, latent_dim=16, context_dim=64)
    model = LatentSDEForecaster(config=cfg)
    model.eval()

    b = 2
    n_samples = 16
    c_wf = torch.randn(b, 500, 1)

    with torch.no_grad():
        out = model.forward_prior(c_wf, num_samples=n_samples)

    assert out.waveform_mean.shape == (b * n_samples, 200, 1)
    samples_0 = out.waveform_mean[:n_samples, :, 0]
    sample_std = samples_0.std(dim=0).mean()
    assert sample_std.item() > 0.0


def test_brownian_path_reproducibility():
    cfg = ModelConfig(num_leads=1, latent_dim=16, context_dim=64)
    model = LatentSDEForecaster(config=cfg)
    model.eval()

    c_wf = torch.randn(1, 500, 1)
    t0 = 0.0
    t1 = 2.0

    bm1 = torchsde.BrownianInterval(t0=t0, t1=t1, size=(1, 16), entropy=42)
    bm2 = torchsde.BrownianInterval(t0=t0, t1=t1, size=(1, 16), entropy=42)
    bm3 = torchsde.BrownianInterval(t0=t0, t1=t1, size=(1, 16), entropy=999)

    # Fixed Brownian path and fixed seed produce same result
    torch.manual_seed(100)
    out1 = model.forward_prior(c_wf, num_samples=1, brownian_motion=bm1)

    torch.manual_seed(100)
    out2 = model.forward_prior(c_wf, num_samples=1, brownian_motion=bm2)

    # Different Brownian path produces different result
    torch.manual_seed(100)
    out3 = model.forward_prior(c_wf, num_samples=1, brownian_motion=bm3)

    assert torch.allclose(out1.latent_path, out2.latent_path, atol=1e-5)
    assert not torch.allclose(out1.latent_path, out3.latent_path, atol=1e-3)
