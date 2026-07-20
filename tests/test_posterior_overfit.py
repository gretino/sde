import pytest
import torch

from ecg_forecast.config import ModelConfig
from ecg_forecast.models import LatentSDEForecaster
from ecg_forecast.losses.elbo import compute_elbo_loss
from ecg_forecast.metrics.waveform import compute_waveform_metrics


def test_posterior_overfit_acceptance_criteria():
    cfg = ModelConfig(num_leads=1, latent_dim=16, context_dim=64)
    model = LatentSDEForecaster(config=cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

    b = 2
    t_ctx = torch.linspace(0, 5, 500).unsqueeze(1)
    t_fut = torch.linspace(5, 7, 200).unsqueeze(1)

    # Smooth sinusoidal waveforms representing ECG rhythm
    c_wf = torch.sin(2 * 3.14159 * 1.2 * t_ctx).unsqueeze(0).repeat(b, 1, 1)
    f_wf = torch.sin(2 * 3.14159 * 1.2 * t_fut).unsqueeze(0).repeat(b, 1, 1)

    for step in range(150):
        optimizer.zero_grad()
        out = model.forward_posterior(c_wf, f_wf)
        loss, _ = compute_elbo_loss(
            pred_mean=out.waveform_mean,
            target=f_wf,
            scale=out.waveform_scale,
            initial_kl=out.initial_kl,
            path_kl=out.path_kl,
            beta_initial=0.0,
            beta_path=0.0,
        )
        loss.backward()
        optimizer.step()

    # Verify Section 10.4 acceptance criteria
    final_out = model.forward_posterior(c_wf, f_wf)
    wf_m = compute_waveform_metrics(final_out.waveform_mean, f_wf)

    assert torch.isfinite(final_out.initial_kl), "Initial KL must be finite"
    assert torch.isfinite(final_out.path_kl), "Path KL must be finite"
    assert wf_m["pearson"] >= 0.95, f"Posterior Pearson {wf_m['pearson']:.4f} < 0.95"
