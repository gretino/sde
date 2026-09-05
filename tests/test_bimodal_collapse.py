import math
import numpy as np
import pytest
import torch
import torch.nn as nn

from ecg_forecast.models.cnsde import ConditionalNeuralSDE
from ecg_forecast.signatures.signature import compute_signature_features, get_signature_dim


def test_synthetic_bimodal_collapse():
    """Section 17 Synthetic Bimodal Collapse Test.

    Identical context x_context = 0.
    Future is randomly either Y+(t) = sin(2*pi*f*t) or Y-(t) = -sin(2*pi*f*t).
    Neural SDE must learn to generate both modes (+sin and -sin) from the same context,
    avoiding collapsing to the conditional mean (0).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    L = 100
    fs = 50.0
    t = torch.linspace(0.0, 2.0, steps=L, device=device)
    sin_pos = torch.sin(2.0 * math.pi * 1.0 * t).unsqueeze(-1)  # [L, 1]
    sin_neg = -sin_pos                                          # [L, 1]

    # Target signature: E[S(Y) | X] = 0.5 * S(Y+) + 0.5 * S(Y-)
    depth = 2
    dyadic_depth = 1
    lead_lag = False

    sig_pos = compute_signature_features(sin_pos.unsqueeze(0), depth=depth, dyadic_depth=dyadic_depth, lead_lag=lead_lag)
    sig_neg = compute_signature_features(sin_neg.unsqueeze(0), depth=depth, dyadic_depth=dyadic_depth, lead_lag=lead_lag)
    target_sig = 0.5 * (sig_pos + sig_neg)  # [1, sig_dim]

    sig_dim = target_sig.shape[-1]
    latent_dim = 16
    context_dim = 16
    initial_noise_dim = 8

    model = ConditionalNeuralSDE(
        sig_dim=sig_dim,
        context_dim=context_dim,
        latent_dim=latent_dim,
        initial_noise_dim=initial_noise_dim,
        num_leads=1,
        drift_hidden=[32, 32],
        diffusion_hidden=[32, 32],
        sigma_min=0.01,
        sigma_max=0.30,
        dt=0.02,
        t_end=2.0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    # Identical zero context
    dummy_sig_x = torch.zeros(1, sig_dim, device=device)
    y0 = torch.zeros(1, 1, 1, device=device)

    # Train for 40 iterations
    model.train()
    for step in range(40):
        optimizer.zero_grad()
        # Sample K=8 futures per step
        wf_samples, _ = model(dummy_sig_x, y0, num_samples=8, use_adjoint=False)
        # wf_samples is [1, 8, 100, 1]
        flat_wf = wf_samples.view(8, -1, 1)

        # Interpolate to L steps if needed
        if flat_wf.shape[1] != L:
            flat_wf = torch.nn.functional.interpolate(flat_wf.permute(0, 2, 1), size=L, mode="linear").permute(0, 2, 1)

        sigs = compute_signature_features(flat_wf, depth=depth, dyadic_depth=dyadic_depth, lead_lag=lead_lag)
        expected_sig = sigs.mean(dim=0, keepdim=True)

        loss = ((expected_sig - target_sig) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Evaluation with K=32 samples
    model.eval()
    with torch.no_grad():
        wf_samples, _ = model(dummy_sig_x, y0, num_samples=32, use_adjoint=False)

    flat_wf = wf_samples.view(32, -1, 1)
    if flat_wf.shape[1] != L:
        flat_wf = torch.nn.functional.interpolate(flat_wf.permute(0, 2, 1), size=L, mode="linear").permute(0, 2, 1)

    samples_np = flat_wf.squeeze(-1).cpu().numpy()  # [32, L]
    target_pos_np = sin_pos.squeeze(-1).cpu().numpy()

    corrs = []
    amplitudes = []
    for k in range(32):
        s = samples_np[k]
        std_s = np.std(s)
        amplitudes.append(std_s)
        if std_s > 1e-4:
            r = np.corrcoef(s, target_pos_np)[0, 1]
            corrs.append(float(r))
        else:
            corrs.append(0.0)

    max_corr = max(corrs)
    min_corr = min(corrs)
    mean_sample_amp = np.mean(amplitudes)
    ensemble_mean_amp = np.std(np.mean(samples_np, axis=0))

    print(f"\nBimodal test: max_corr={max_corr:.3f}, min_corr={min_corr:.3f}")
    print(f"Mean sample amp={mean_sample_amp:.4f}, Ensemble mean amp={ensemble_mean_amp:.4f}")

    # Verify both positive and negative correlations exist (both modes generated)
    assert max_corr > 0.3, f"Expected positive mode generation, but max_corr={max_corr}"
    assert min_corr < -0.3, f"Expected negative mode generation, but min_corr={min_corr}"
    # Verify ensemble mean amplitude is lower than individual sample amplitudes
    assert ensemble_mean_amp < mean_sample_amp
