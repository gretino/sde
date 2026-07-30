#!/usr/bin/env python3
"""Script 16: diagnose_uncertainty_sources.py - Multisample Uncertainty Diagnostic Tool (Gate 6)."""

import os
import argparse
import json
import numpy as np
import torch

from ecg_forecast.config import Config
from ecg_forecast.data.incart import get_incart_dataloaders
from ecg_forecast.debug.checkpoint_loader import load_forecaster_checkpoint
from ecg_forecast.debug.metrics import compute_uncertainty_debug_metrics
from ecg_forecast.debug.reporting import save_debug_artifacts


def main():
    parser = argparse.ArgumentParser(description="Run uncertainty source diagnostics (Script 16).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/incart")
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Diagnose Uncertainty] Loading checkpoint: {args.checkpoint}")
    model, cfg = load_forecaster_checkpoint(args.checkpoint, device=device)

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Dataset directory '{args.data_dir}' does not exist. Cannot run uncertainty diagnostics.")

    cfg.data.data_dir = args.data_dir
    _, val_loader, _, _ = get_incart_dataloaders(config=cfg.data, batch_size=1, num_workers=0)

    checkpoint_name = os.path.basename(args.checkpoint).replace(".pt", "")
    if args.output_dir is None:
        args.output_dir = os.path.join("artifacts/debug/diagnose_uncertainty_sources", checkpoint_name)

    val_batch = next(iter(val_loader))
    c_wf = val_batch["context_waveform"].to(device)  # [1, 500, 12]
    f_wf = val_batch["future_waveform"].to(device)   # [1, 200, 12]

    # Context encoding
    c_summary, _, prior_mean, prior_logvar = model.context_encoder(c_wf)

    prior_logvar_stats = {
        "mean": float(prior_logvar.mean().item()),
        "min": float(prior_logvar.min().item()),
        "max": float(prior_logvar.max().item()),
        "fraction_at_lower_clamp": float((prior_logvar <= -7.99).float().mean().item()),
    }

    n_samples = args.num_samples
    target_samples = f_wf.size(1)
    
    from ecg_forecast.utils.timegrid import make_latent_times
    ts = make_latent_times(future_samples=target_samples, sampling_rate=100, latent_rate=25, device=device)

    # Expanded context summary for n_samples
    c_summary_rep = c_summary.repeat(n_samples, 1)
    p_mean_rep = prior_mean.repeat(n_samples, 1)
    p_logvar_rep = prior_logvar.repeat(n_samples, 1)

    # 1. Initial-state variation only (sample z0, deterministic=True, no diffusion)
    z0_init_only = model._reparameterize(p_mean_rep, p_logvar_rep)
    with torch.no_grad():
        lat_init_only, _ = model.sde.integrate(
            z0=z0_init_only, ts=ts, context_summary=c_summary_rep, mode="prior", deterministic=True
        )
        wf_init_only, _ = model.decoder(lat_init_only, c_summary_rep, target_len=target_samples)

    m_init_only = compute_uncertainty_debug_metrics(
        samples_waveform=wf_init_only,
        target_waveform=f_wf[0],
        samples_latent=lat_init_only,
    )

    # 2. Brownian variation only (z0 = exact mean, stochastic SDE, deterministic=False)
    z0_det = p_mean_rep
    with torch.no_grad():
        lat_brownian_only, _ = model.sde.integrate(
            z0=z0_det, ts=ts, context_summary=c_summary_rep, mode="prior", deterministic=False
        )
        wf_brownian_only, _ = model.decoder(lat_brownian_only, c_summary_rep, target_len=target_samples)

    m_brownian_only = compute_uncertainty_debug_metrics(
        samples_waveform=wf_brownian_only,
        target_waveform=f_wf[0],
        samples_latent=lat_brownian_only,
    )

    # 3. Combined variation (sample z0, stochastic SDE, deterministic=False)
    with torch.no_grad():
        lat_combined, _ = model.sde.integrate(
            z0=z0_init_only, ts=ts, context_summary=c_summary_rep, mode="prior", deterministic=False
        )
        wf_combined, _ = model.decoder(lat_combined, c_summary_rep, target_len=target_samples)

    m_combined = compute_uncertainty_debug_metrics(
        samples_waveform=wf_combined,
        target_waveform=f_wf[0],
        samples_latent=lat_combined,
    )

    # Interpretation
    interpretation = []
    if prior_logvar_stats["fraction_at_lower_clamp"] > 0.50:
        interpretation.append("CRITICAL: prior log-variance has collapsed to lower clamp (-8.0)")
    if m_init_only["latent_variance_retention"] < 0.10:
        interpretation.append("prior drift is strongly contracting (erases initial sample diversity)")
    if m_combined["latent_zT_sample_var"] > 0.05 and m_combined["mean_waveform_sample_std"] < 0.01:
        interpretation.append("decoder is insensitive to latent sample variation")

    summary_data = {
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "num_samples": n_samples,
        "prior_logvar_stats": prior_logvar_stats,
        "initial_state_variation_only": m_init_only,
        "brownian_variation_only": m_brownian_only,
        "combined_variation": m_combined,
        "interpretation": interpretation,
    }

    save_debug_artifacts(
        output_dir=args.output_dir,
        summary_data=summary_data,
        config=cfg,
    )

    print(f"[Diagnose Uncertainty] Complete. Artifacts written to {args.output_dir}")
    print("Interpretation:", interpretation)


if __name__ == "__main__":
    main()
