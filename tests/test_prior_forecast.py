import pytest
import torch

from ecg_forecast.config import ModelConfig
from ecg_forecast.models import LatentSDEForecaster


def test_prior_forecast_properties():
    cfg = ModelConfig(num_leads=12, latent_dim=32, context_dim=128)
    model = LatentSDEForecaster(config=cfg)
    model.eval()

    b = 3
    c_wf = torch.randn(b, 500, 12)

    with torch.no_grad():
        out = model.forward_prior(c_wf, num_samples=1)

    assert out.waveform_mean.shape == (b, 200, 12)
    assert not torch.isnan(out.waveform_mean).any()

    # Prior output is nonconstant over time
    output_std = out.waveform_mean.std(dim=1).mean()
    assert output_std.item() > 1e-4, "Prior output waveform is flat/constant"

    # Prior latent temporal standard deviation is nonzero
    latent_std = out.latent_path.std(dim=1).mean()
    assert latent_std.item() > 1e-4, "Prior latent trajectory is flat/constant"
