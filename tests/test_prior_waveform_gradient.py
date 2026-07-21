import pytest
import torch
from ecg_forecast.config import ModelConfig
from ecg_forecast.models import LatentSDEForecaster
from ecg_forecast.losses.elbo import compute_laplace_nll


def test_prior_waveform_loss_gradients():
    cfg = ModelConfig(num_leads=1, latent_dim=16, context_dim=64)
    model = LatentSDEForecaster(config=cfg)

    c_wf = torch.randn(2, 500, 1)
    f_wf = torch.randn(2, 200, 1)

    prior_out = model.forward_prior(c_wf)
    prior_nll = compute_laplace_nll(prior_out.waveform_mean, f_wf, prior_out.waveform_scale)

    prior_nll.backward()

    # Verify prior initial heads and prior drift get non-zero gradients from prior waveform loss
    prior_drift_grad = model.sde.sde_func.prior_drift_net[0].weight.grad
    assert prior_drift_grad is not None and torch.norm(prior_drift_grad) > 0.0, "Prior drift must receive non-zero gradient from prior waveform loss"

    prior_mean_grad = model.context_encoder.fc_mean.weight.grad
    assert prior_mean_grad is not None and torch.norm(prior_mean_grad) > 0.0, "Prior mean head must receive non-zero gradient from prior waveform loss"
