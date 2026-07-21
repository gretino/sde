#!/usr/bin/env python3
"""Script 21: compare_stage_checkpoints.py - Benchmark Comparison of Checkpoints."""

import os
import argparse
import numpy as np
import torch

from ecg_forecast.config import Config
from ecg_forecast.data.incart import get_incart_dataloaders
from ecg_forecast.debug.checkpoint_loader import load_forecaster_checkpoint
from ecg_forecast.debug.rollouts import run_decomposed_rollouts
from ecg_forecast.debug.metrics import (
    compute_waveform_debug_metrics,
    compute_rhythm_debug_metrics,
    compute_latent_debug_metrics,
    compute_uncertainty_debug_metrics,
)
from ecg_forecast.debug.reporting import save_debug_artifacts


def evaluate_checkpoint(ckpt_path: str, val_batch: dict, device: str) -> dict:
    model, cfg = load_forecaster_checkpoint(ckpt_path, device=device)

    c_wf = val_batch["context_waveform"].to(device)
    f_wf = val_batch["future_waveform"].to(device)

    rollouts = run_decomposed_rollouts(model, c_wf, f_wf)

    wf_A = rollouts["A"]["waveform_mean"]  # Posterior reconstruction
    wf_B = rollouts["B"]["waveform_mean"]  # Deterministic prior forecast

    m_A_wf = compute_waveform_debug_metrics(wf_A, f_wf)
    m_B_wf = compute_waveform_debug_metrics(wf_B, f_wf)
    m_B_rh = compute_rhythm_debug_metrics(wf_B, f_wf)
    m_B_lat = compute_latent_debug_metrics(rollouts["B"]["latent_path"])

    # 128-sample prior forecast
    with torch.no_grad():
        c_summary, _, prior_mean, prior_logvar = model.context_encoder(c_wf)
        ts = torch.linspace(0.04, 2.0, 50, device=device)

        num_samples = 128
        c_summary_rep = c_summary.repeat(num_samples, 1)
        prior_mean_rep = prior_mean.repeat(num_samples, 1)
        prior_logvar_rep = prior_logvar.repeat(num_samples, 1)

        z0_samples = model._reparameterize(prior_mean_rep, prior_logvar_rep)
        lat_samples, _ = model.sde.integrate(
            z0=z0_samples, ts=ts, context_summary=c_summary_rep, mode="prior"
        )
        wf_samples, _ = model.decoder(lat_samples, c_summary_rep)

    m_unc = compute_uncertainty_debug_metrics(
        samples_waveform=wf_samples,
        target_waveform=f_wf[0],
        samples_latent=lat_samples,
    )

    return {
        "checkpoint": ckpt_path,
        "posterior_reconstruction_pearson": m_A_wf["macro_pearson"],
        "posterior_reconstruction_mse": m_A_wf["mse"],
        "prior_forecast_pearson": m_B_wf["macro_pearson"],
        "prior_forecast_mse": m_B_wf["mse"],
        "prior_rpeak_f1": m_B_rh["rpeak_f1"],
        "prior_heart_rate_mae": m_B_rh["heart_rate_mae"],
        "latent_temporal_std": m_B_lat["latent_temporal_std"],
        "latent_path_length": m_B_lat["latent_path_length"],
        "uncertainty_90_coverage": m_unc["interval_coverage_90"],
        "uncertainty_90_width": m_unc["interval_width_90"],
        "sample_waveform_std": m_unc["mean_waveform_sample_std"],
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Stage A, B, C checkpoints (Script 21).")
    parser.add_argument("--stage_a_ckpt", type=str, required=True, help="Stage A checkpoint path")
    parser.add_argument("--stage_b_ckpt", type=str, default=None, help="Stage B checkpoint path")
    parser.add_argument("--stage_c_ckpt", type=str, default=None, help="Stage C checkpoint path")
    parser.add_argument("--data_dir", type=str, default="data/incart")
    parser.add_argument("--output_dir", type=str, default="artifacts/debug/compare_stage_checkpoints/benchmark")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Dataset directory '{args.data_dir}' does not exist. Cannot run checkpoint comparison.")

    cfg = Config()
    cfg.data.data_dir = args.data_dir

    print(f"[Compare Checkpoints] Loading validation dataset from {args.data_dir}...")
    _, val_loader, _, _ = get_incart_dataloaders(config=cfg.data, batch_size=1, num_workers=0)
    val_batch = next(iter(val_loader))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoints_to_eval = [("Stage_A", args.stage_a_ckpt)]
    if args.stage_b_ckpt:
        checkpoints_to_eval.append(("Stage_B", args.stage_b_ckpt))
    if args.stage_c_ckpt:
        checkpoints_to_eval.append(("Stage_C", args.stage_c_ckpt))

    results = {}
    print("[Compare Checkpoints] Benchmarking checkpoints on identical validation sample...")

    for stage_label, ckpt_path in checkpoints_to_eval:
        print(f"  Evaluating {stage_label}: {ckpt_path}")
        res = evaluate_checkpoint(ckpt_path, val_batch, device=device)
        results[stage_label] = res

    summary_data = {
        "stage_comparison": results,
    }

    save_debug_artifacts(
        output_dir=args.output_dir,
        summary_data=summary_data,
        config=cfg,
    )

    print(f"[Compare Checkpoints] Complete. Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
