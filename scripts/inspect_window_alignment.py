#!/usr/bin/env python3
"""Script 20: inspect_window_alignment.py - Gate 0 Data Integrity Checker."""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from ecg_forecast.config import Config
from ecg_forecast.data.incart import get_incart_dataloaders
from ecg_forecast.debug.reporting import save_debug_artifacts


def main():
    parser = argparse.ArgumentParser(description="Inspect ECG window alignment and data integrity (Gate 0).")
    parser.add_argument("--data_dir", type=str, default="data/incart")
    parser.add_argument("--cache_dir", type=str, default="cache/preprocessed")
    parser.add_argument("--output_dir", type=str, default="artifacts/debug/inspect_window_alignment/incart")
    parser.add_argument("--num_samples_to_inspect", type=int, default=32)
    args = parser.parse_args()

    cfg = Config()
    cfg.data.data_dir = args.data_dir
    cfg.data.cache_dir = args.cache_dir

    if not os.path.exists(args.data_dir) and not os.path.exists(args.cache_dir):
        raise FileNotFoundError(
            f"Dataset directory '{args.data_dir}' or cache directory '{args.cache_dir}' does not exist. "
            f"Cannot inspect dataset window alignment without raw or preprocessed data."
        )

    print(f"[Inspect Data] Loading dataloaders from {args.data_dir}...")
    try:
        train_loader, val_loader, test_loader, splits = get_incart_dataloaders(
            config=cfg.data,
            batch_size=16,
            num_workers=0,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from '{args.data_dir}': {e}") from e

    boundary_jumps_per_lead = []
    context_stds_per_lead = []
    future_stds_per_lead = []
    future_rpeak_counts = []
    near_constant_windows = 0
    no_rpeaks_count = 0

    inspected = 0
    per_sample_rows = []

    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)

    for batch in val_loader:
        if inspected >= args.num_samples_to_inspect:
            break

        c_wf = batch["context_waveform"]  # [B, 500, 12]
        f_wf = batch["future_waveform"]   # [B, 200, 12]
        c_times = batch["context_times"]
        f_times = batch["future_times"]
        r_peaks = batch.get("future_r_peaks", [])

        b = c_wf.size(0)
        for i in range(b):
            if inspected >= args.num_samples_to_inspect:
                break

            rec_id = batch["record_id"][i] if "record_id" in batch else f"rec_{inspected}"
            w_start = batch["window_start"][i] if "window_start" in batch else inspected

            # Boundary jump at t=0 between last context step and first future step
            last_ctx = c_wf[i, -1, :]  # [12]
            first_fut = f_wf[i, 0, :]   # [12]
            jumps = torch.abs(first_fut - last_ctx).numpy()
            boundary_jumps_per_lead.append(jumps)

            ctx_std = c_wf[i].std(dim=0).numpy()
            fut_std = f_wf[i].std(dim=0).numpy()
            context_stds_per_lead.append(ctx_std)
            future_stds_per_lead.append(fut_std)

            # Check near-constant window
            if float(np.mean(ctx_std)) < 1e-4 or float(np.mean(fut_std)) < 1e-4:
                near_constant_windows += 1

            # R-peaks count
            num_peaks = len(r_peaks[i]) if i < len(r_peaks) else 0
            future_rpeak_counts.append(num_peaks)
            if num_peaks == 0:
                no_rpeaks_count += 1

            per_sample_rows.append({
                "sample_idx": inspected,
                "record_id": str(rec_id),
                "window_start": float(w_start),
                "mean_boundary_jump": float(np.mean(jumps)),
                "max_boundary_jump": float(np.max(jumps)),
                "context_mean_std": float(np.mean(ctx_std)),
                "future_mean_std": float(np.mean(fut_std)),
                "num_future_rpeaks": num_peaks,
            })

            # Plot first 3 samples
            if inspected < 3:
                fig, axes = plt.subplots(4, 3, figsize=(15, 10), sharex=True)
                axes = axes.flatten()
                lead_names = cfg.data.lead_names if hasattr(cfg.data, "lead_names") else [f"Lead {k+1}" for k in range(12)]
                
                t_c = c_times[i].numpy() if c_times.dim() == 2 else c_times.numpy()
                t_f = f_times[i].numpy() if f_times.dim() == 2 else f_times.numpy()

                for l_idx in range(12):
                    ax = axes[l_idx]
                    ax.plot(t_c, c_wf[i, :, l_idx].numpy(), label="Context (-5 to 0s)", color="blue")
                    ax.plot(t_f, f_wf[i, :, l_idx].numpy(), label="Future (0 to 2s)", color="green")
                    ax.axvline(x=0.0, color="red", linestyle="--", alpha=0.7, label="Anchor (t=0)")
                    ax.set_title(lead_names[l_idx] if l_idx < len(lead_names) else f"Lead {l_idx+1}")
                    if l_idx == 0:
                        ax.legend(loc="upper left", fontsize=8)
                plt.tight_layout()
                fig.savefig(os.path.join(args.output_dir, "plots", f"window_inspection_sample_{inspected}.png"))
                plt.close(fig)


            inspected += 1

    boundary_jumps_per_lead = np.array(boundary_jumps_per_lead)  # [N, 12]
    context_stds_per_lead = np.array(context_stds_per_lead)      # [N, 12]
    future_stds_per_lead = np.array(future_stds_per_lead)        # [N, 12]

    summary = {
        "num_inspected_samples": inspected,
        "median_boundary_jump_overall": float(np.median(boundary_jumps_per_lead)),
        "p99_boundary_jump_overall": float(np.percentile(boundary_jumps_per_lead, 99)),
        "mean_context_std": float(np.mean(context_stds_per_lead)),
        "mean_future_std": float(np.mean(future_stds_per_lead)),
        "pct_near_constant_windows": float((near_constant_windows / inspected) * 100.0),
        "pct_no_rpeak_windows": float((no_rpeaks_count / inspected) * 100.0),
        "rpeak_count_distribution": {
            "mean": float(np.mean(future_rpeak_counts)),
            "min": int(np.min(future_rpeak_counts)) if len(future_rpeak_counts) > 0 else 0,
            "max": int(np.max(future_rpeak_counts)) if len(future_rpeak_counts) > 0 else 0,
        },
        "gate_0_passed": bool(np.median(boundary_jumps_per_lead) < 0.2 and near_constant_windows == 0),
    }

    save_debug_artifacts(
        output_dir=args.output_dir,
        summary_data=summary,
        per_sample_rows=per_sample_rows,
        config=cfg,
    )

    print(f"[Inspect Data] Complete. Gate 0 Passed: {summary['gate_0_passed']}. Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
