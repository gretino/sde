from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from ..models.latent_sde_forecaster import LatentSDEForecaster


def run_decomposed_rollouts(
    model: LatentSDEForecaster,
    context_waveform: torch.Tensor,
    future_waveform: torch.Tensor,
    context_times: Optional[torch.Tensor] = None,
    future_times: Optional[torch.Tensor] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Runs deterministic zero-diffusion decomposed rollouts A, B, C, D, E for diagnostics.
    
    Returns a dictionary mapping rollout key ('A', 'B', 'C', 'D', 'E') to a dict containing:
        - 'waveform_mean': [B, T_waveform, num_leads]
        - 'latent_path': [B, T_latent, latent_dim]
    """
    model.eval()
    device = context_waveform.device
    b = context_waveform.size(0)

    # 1. Prior Context Encoding -> mu_p, c_summary
    c_summary, c_tokens, prior_mean, prior_logvar = model.context_encoder(context_waveform)

    # 2. Posterior Encoding -> mu_q, rec_path
    full_wf = torch.cat([context_waveform, future_waveform], dim=1)
    post_summary, rec_path, post_mean, post_logvar = model.posterior_encoder(full_wf, c_summary)

    # Timestamps [50] (0.04, 0.08, ..., 2.00)
    if future_times is not None and future_times.dim() == 1:
        ts = future_times[::4]
    elif future_times is not None and future_times.dim() == 2:
        ts = future_times[0, ::4]
    else:
        ts = torch.linspace(0.04, 2.0, 50, device=device)

    # Deterministic z0s (mean only, zero diffusion)
    z0_p = prior_mean
    z0_q = post_mean

    t_target = future_waveform.size(1)
    results = {}

    with torch.no_grad():
        # --- Rollout A: Full posterior reconstruction ---
        latent_A, _ = model.sde.integrate(
            z0=z0_q,
            ts=ts,
            context_summary=c_summary,
            recognition_path=rec_path,
            mode="posterior",
            brownian_motion=None,
        )
        wf_A, _ = model.decoder(latent_A, c_summary, target_len=t_target)
        results["A"] = {"waveform_mean": wf_A, "latent_path": latent_A}

        # --- Rollout B: Full prior forecast ---
        latent_B, _ = model.sde.integrate(
            z0=z0_p,
            ts=ts,
            context_summary=c_summary,
            mode="prior",
            brownian_motion=None,
        )
        wf_B, _ = model.decoder(latent_B, c_summary, target_len=t_target)
        results["B"] = {"waveform_mean": wf_B, "latent_path": latent_B}

        # --- Rollout C: Oracle initial-state rollout (mu_q + prior drift) ---
        latent_C, _ = model.sde.integrate(
            z0=z0_q,
            ts=ts,
            context_summary=c_summary,
            mode="prior",
            brownian_motion=None,
        )
        wf_C, _ = model.decoder(latent_C, c_summary, target_len=t_target)
        results["C"] = {"waveform_mean": wf_C, "latent_path": latent_C}

        # --- Rollout D: Oracle future-dynamics rollout (mu_p + posterior drift) ---
        latent_D, _ = model.sde.integrate(
            z0=z0_p,
            ts=ts,
            context_summary=c_summary,
            recognition_path=rec_path,
            mode="posterior",
            brownian_motion=None,
        )
        wf_D, _ = model.decoder(latent_D, c_summary, target_len=t_target)
        results["D"] = {"waveform_mean": wf_D, "latent_path": latent_D}

        # --- Rollout E: Direct teacher latent decoding ---
        wf_E, _ = model.decoder(latent_A, c_summary, target_len=t_target)
        results["E"] = {"waveform_mean": wf_E, "latent_path": latent_A}


    return results
