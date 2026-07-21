import pytest
import torch
from ecg_forecast.config import ModelConfig
from ecg_forecast.models import LatentSDEForecaster
from ecg_forecast.losses.elbo import compute_initial_teacher_loss, compute_drift_teacher_loss


def test_teacher_detach_no_gradient_path_to_posterior():
    cfg = ModelConfig(num_leads=1, latent_dim=16, context_dim=64)
    model = LatentSDEForecaster(config=cfg)

    c_wf = torch.randn(2, 500, 1)
    f_wf = torch.randn(2, 200, 1)

    # Compute detached teacher outputs
    with torch.no_grad():
        c_summary_t, _, _, _ = model.context_encoder(c_wf)
        full_wf = torch.cat([c_wf, f_wf], dim=1)
        _, _, post_mean_det, post_logvar_det = model.posterior_encoder(full_wf, c_summary_t)

    # Compute student prior outputs
    c_summary, _, prior_mean, prior_logvar = model.context_encoder(c_wf)

    init_teacher_loss = compute_initial_teacher_loss(prior_mean, prior_logvar, post_mean_det, post_logvar_det)
    init_teacher_loss.backward()

    # Verify that posterior encoder receives ZERO gradient from initial teacher loss
    for p in model.posterior_encoder.parameters():
        assert p.grad is None or (p.grad == 0).all(), "Posterior encoder must have zero gradient from teacher loss"
