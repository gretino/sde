#!/usr/bin/env python3
"""Script 14: diagnose_rollout_components.py - Decomposed Rollouts Diagnostic Tool."""

import os
import argparse
import json
import torch
import numpy as np

from ecg_forecast.config import Config
from ecg_forecast.data.incart import get_incart_dataloaders
from ecg_forecast.debug.checkpoint_loader import load_forecaster_checkpoint
from ecg_forecast.debug.rollouts import run_decomposed_rollouts
from ecg_forecast.debug.metrics import (
    compute_waveform_debug_metrics,
    compute_rhythm_debug_metrics,
    compute_latent_debug_metrics,
)
from ecg_forecast.debug.reporting import save_debug_artifacts


def evaluate_rollout_at_horizons(
    wf_pred: torch.Tensor,
    wf_target: torch.Tensor,
    latent_path: torch.Tensor,
    horizons_sec: list = [0.25, 0.5, 1.0, 2.0],
    sampling_rate: int = 100,
) -> dict:
    results = {}
    for h_sec in horizons_sec:
        t_samples = int(round(h_sec * sampling_rate))
        t_latent = int(round(h_sec * 25))

        pred_sub = wf_pred[:, :t_samples, :]
        target_sub = wf_target[:, :t_samples, :]
        latent_sub = latent_path[:, :t_latent, :]

        wf_m = compute_waveform_debug_metrics(pred_sub, target_sub)
        rhythm_m = compute_rhythm_debug_metrics(pred_sub, target_sub, sampling_rate=sampling_rate)
        latent_m = compute_latent_debug_metrics(latent_sub)

        results[f"{h_sec}s"] = {
            "mse": wf_m["mse"],
            "mae": wf_m["mae"],
            "macro_pearson": wf_m["macro_pearson"],
            "rpeak_f1": rhythm_m["rpeak_f1"],
            "heart_rate_mae": rhythm_m["heart_rate_mae"],
            "next_rpeak_timing_mae_ms": rhythm_m["next_rpeak_timing_mae_ms"],
            "waveform_temporal_std": wf_m["waveform_temporal_std"],
            "latent_temporal_std": latent_m["latent_temporal_std"],
            "latent_path_length": latent_m["latent_path_length"],
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Run decomposed rollouts diagnostic (Script 14).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/incart")
    parser.add_argument("--num_batches", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Diagnose Rollouts] Loading checkpoint: {args.checkpoint}")
    model, cfg = load_forecaster_checkpoint(args.checkpoint, device=device)

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Dataset directory '{args.data_dir}' does not exist. Cannot run rollout diagnostics.")

    cfg.data.data_dir = args.data_dir
    _, val_loader, _, _ = get_incart_dataloaders(config=cfg.data, batch_size=args.batch_size, num_workers=0)

    checkpoint_name = os.path.basename(args.checkpoint).replace(".pt", "")
    if args.output_dir is None:
        args.output_dir = os.path.join("artifacts/debug/diagnose_rollout_components", checkpoint_name)

    rollout_types = ["A", "B", "C", "D", "E"]
    accumulated = {r: {h: [] for h in ["0.25s", "0.5s", "1.0s", "2.0s"]} for r in rollout_types}
    per_sample_rows = []

    print("[Diagnose Rollouts] Running rollouts A, B, C, D, E...")
    batch_cnt = 0
    with torch.no_grad():
        for batch in val_loader:
            if batch_cnt >= args.num_batches:
                break

            c_wf = batch["context_waveform"].to(device)
            f_wf = batch["future_waveform"].to(device)

            rollouts = run_decomposed_rollouts(model, c_wf, f_wf)

            for r_key in rollout_types:
                wf_pred = rollouts[r_key]["waveform_mean"]
                latent_p = rollouts[r_key]["latent_path"]

                h_metrics = evaluate_rollout_at_horizons(wf_pred, f_wf, latent_p)
                for h_sec, m_dict in h_metrics.items():
                    accumulated[r_key][h_sec].append(m_dict)

            # Record per-sample row for 0.5s horizon on rollout B vs A
            b_size = c_wf.size(0)
            for i in range(b_size):
                rec_id = batch["record_id"][i] if "record_id" in batch else f"batch{batch_cnt}_sample{i}"
                w_start = batch["window_start"][i] if "window_start" in batch else i

                wf_A_sub = rollouts["A"]["waveform_mean"][i:i+1, :50, :]
                wf_B_sub = rollouts["B"]["waveform_mean"][i:i+1, :50, :]
                f_sub = f_wf[i:i+1, :50, :]

                m_A = compute_waveform_debug_metrics(wf_A_sub, f_sub)
                m_B = compute_waveform_debug_metrics(wf_B_sub, f_sub)
                r_B = compute_rhythm_debug_metrics(wf_B_sub, f_sub)

                per_sample_rows.append({
                    "record_id": str(rec_id),
                    "window_start": float(w_start),
                    "forecast_horizon": 0.5,
                    "posterior_rec_mse": m_A["mse"],
                    "prior_forecast_mse": m_B["mse"],
                    "posterior_rec_pearson": m_A["macro_pearson"],
                    "prior_forecast_pearson": m_B["macro_pearson"],
                    "prior_forecast_rpeak_f1": r_B["rpeak_f1"],
                })

            batch_cnt += 1

    # Aggregate results across batches
    summary_results = {}
    for r_key in rollout_types:
        summary_results[r_key] = {}
        for h_sec in ["0.25s", "0.5s", "1.0s", "2.0s"]:
            dicts = accumulated[r_key][h_sec]
            summary_results[r_key][h_sec] = {
                k: float(np.mean([d[k] for d in dicts])) for k in dicts[0].keys()
            }

    # Interpretation logic for 0.5s horizon
    b_pearson = summary_results["B"]["0.5s"]["macro_pearson"]
    c_pearson = summary_results["C"]["0.5s"]["macro_pearson"]
    d_pearson = summary_results["D"]["0.5s"]["macro_pearson"]
    a_pearson = summary_results["A"]["0.5s"]["macro_pearson"]

    interpretation = []
    if c_pearson >= 0.70 and b_pearson < 0.50:
        interpretation.append("prior initial-state encoder is the main failure mode")
    if c_pearson < 0.50:
        interpretation.append("prior autonomous drift is the main failure mode")
    if d_pearson >= 0.70 and b_pearson < 0.50:
        interpretation.append("prior drift and/or prior initial state are failing")
    if a_pearson >= 0.80 and b_pearson < 0.50 and c_pearson < 0.50:
        interpretation.append("future-conditioned teacher is valid but context-only prior is not learnable in current form")

    summary_data = {
        "checkpoint": args.checkpoint,
        "rollouts": summary_results,
        "interpretation": interpretation,
    }

    save_debug_artifacts(
        output_dir=args.output_dir,
        summary_data=summary_data,
        per_sample_rows=per_sample_rows,
        config=cfg,
    )

    print(f"[Diagnose Rollouts] Complete. Summary written to {args.output_dir}/summary.json")
    print("Diagnosis Interpretation:", interpretation)


if __name__ == "__main__":
    main()
