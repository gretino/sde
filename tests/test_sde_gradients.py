import pytest
import torch

from ecg_forecast.config import ModelConfig
from ecg_forecast.models import LatentSDEForecaster
from ecg_forecast.losses.elbo import compute_elbo_loss
from ecg_forecast.losses.morphology import compute_morphology_loss


def test_sde_gradients_all_modules():
    cfg = ModelConfig(num_leads=1, latent_dim=16, context_dim=64)
    model = LatentSDEForecaster(config=cfg)
    model.set_stage("C")

    b = 2
    c_wf = torch.randn(b, 500, 1, requires_grad=True)
    f_wf = torch.randn(2, 200, 1, requires_grad=True)

    post_out = model.forward_posterior(c_wf, f_wf)

    elbo, _ = compute_elbo_loss(
        pred_mean=post_out.waveform_mean,
        target=f_wf,
        scale=post_out.waveform_scale,
        initial_kl=post_out.initial_kl,
        path_kl=post_out.path_kl,
        beta_initial=1.0,
        beta_path=1.0,
    )
    morph, _ = compute_morphology_loss(pred=post_out.waveform_mean, target=f_wf)

    loss = elbo + morph
    loss.backward()

    # Check key module parameter gradients explicitly
    modules_to_check = {
        "context_encoder": model.context_encoder.conv1.weight,
        "context_prior_head_mean": model.context_encoder.fc_mean.weight,
        "context_prior_head_logvar": model.context_encoder.fc_logvar.weight,
        "posterior_encoder": model.posterior_encoder.conv1.weight,
        "posterior_head_mean": model.posterior_encoder.fc_mean.weight,
        "posterior_head_logvar": model.posterior_encoder.fc_logvar.weight,
        "prior_drift": model.sde.sde_func.prior_drift_net[0].weight,
        "posterior_drift": model.sde.sde_func.posterior_drift_net[0].weight,
        "diffusion": model.sde.sde_func.raw_sigma,
        "decoder_net": model.decoder.net[0].weight,
        "observation_scale": model.decoder.raw_obs_log_scale,
    }

    for name, param in modules_to_check.items():
        assert param.grad is not None, f"Gradient for {name} is None"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
        assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
        assert torch.norm(param.grad) > 0.0, f"Zero gradient in {name}"
