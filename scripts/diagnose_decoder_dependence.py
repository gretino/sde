#!/usr/bin/env python3
"""Script 15: diagnose_decoder_dependence.py - Decoder Sensitivity & Latent Utilization Diagnostic (Gate 4)."""

import os
import argparse
import json
import numpy as np
import torch

from ecg_forecast.config import Config
from ecg_forecast.data.incart import get_incart_dataloaders
from ecg_forecast.debug.checkpoint_loader import load_forecaster_checkpoint
from ecg_forecast.debug.metrics import (
    compute_waveform_debug_metrics,
    compute_rhythm_debug_metrics,
)
from ecg_forecast.debug.reporting import save_debug_artifacts


def main():
    parser = argparse.ArgumentParser(description="Run decoder dependence diagnostics (Script 15).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--data_dir", type=str, default="data/incart")
    parser.add_argument("--num_batches", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Diagnose Decoder] Loading checkpoint: {args.checkpoint}")
    model, cfg = load_forecaster_checkpoint(args.checkpoint, device=device)

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Dataset directory '{args.data_dir}' does not exist. Cannot run decoder diagnostics.")

    cfg.data.data_dir = args.data_dir
    _, val_loader, _, _ = get_incart_dataloaders(config=cfg.data, batch_size=args.batch_size, num_workers=0)

    checkpoint_name = os.path.basename(args.checkpoint).replace(".pt", "")
    if args.output_dir is None:
        args.output_dir = os.path.join("artifacts/debug/diagnose_decoder_dependence", checkpoint_name)

    ablations = [
        "posterior_latent_plus_correct_context",
        "posterior_latent_plus_zero_context",
        "zero_latent_plus_correct_context",
        "time_shuffled_latent_plus_correct_context",
        "batch_shuffled_latent_plus_correct_context",
        "posterior_latent_plus_batch_shuffled_context",
        "prior_latent_plus_correct_context",
    ]

    ablation_metrics = {k: [] for k in ablations}

    from ecg_forecast.utils.timegrid import make_latent_times

    print("[Diagnose Decoder] Running deterministic decoder ablations...")

    with torch.no_grad():
        for batch_cnt, batch in enumerate(val_loader):
            if batch_cnt >= args.num_batches:
                break

            c_wf = batch["context_waveform"].to(device)
            f_wf = batch["future_waveform"].to(device)
            b = c_wf.size(0)
            target_samples = f_wf.size(1)

            c_summary, _, prior_mean, _ = model.context_encoder(c_wf)
            full_wf = torch.cat([c_wf, f_wf], dim=1)
            _, rec_path, post_mean, _ = model.posterior_encoder(
                full_wf, c_summary, future_samples=target_samples, sampling_rate=100, latent_rate=25
            )

            ts = make_latent_times(future_samples=target_samples, sampling_rate=100, latent_rate=25, device=device)
            
            post_latent, _ = model.sde.integrate(
                z0=post_mean, ts=ts, context_summary=c_summary, recognition_path=rec_path, mode="posterior", deterministic=True
            )
            prior_latent, _ = model.sde.integrate(
                z0=prior_mean, ts=ts, context_summary=c_summary, mode="prior", deterministic=True
            )

            t_target = target_samples

            # A1: posterior_latent + correct_context
            wf1, _ = model.decoder(post_latent, c_summary, target_len=t_target)

            # A2: posterior_latent + zero_context
            wf2, _ = model.decoder(post_latent, torch.zeros_like(c_summary), target_len=t_target)

            # A3: zero_latent + correct_context
            wf3, _ = model.decoder(torch.zeros_like(post_latent), c_summary, target_len=t_target)

            # A4: time_shuffled_posterior_latent + correct_context
            perm_t = torch.randperm(post_latent.size(1))
            wf4, _ = model.decoder(post_latent[:, perm_t, :], c_summary, target_len=t_target)

            # A5: batch_shuffled_posterior_latent + correct_context
            perm_b = torch.randperm(b)
            wf5, _ = model.decoder(post_latent[perm_b], c_summary, target_len=t_target)

            # A6: posterior_latent + batch_shuffled_context
            wf6, _ = model.decoder(post_latent, c_summary[perm_b], target_len=t_target)

            # A7: prior_latent + correct_context
            wf7, _ = model.decoder(prior_latent, c_summary, target_len=t_target)

            abl_preds = [wf1, wf2, wf3, wf4, wf5, wf6, wf7]
            for idx, abl_name in enumerate(ablations):
                w_m = compute_waveform_debug_metrics(abl_preds[idx], f_wf)
                r_m = compute_rhythm_debug_metrics(abl_preds[idx], f_wf)
                ablation_metrics[abl_name].append({
                    "mse": w_m["mse"],
                    "mae": w_m["mae"],
                    "macro_pearson": w_m["macro_pearson"],
                    "rpeak_f1": r_m["rpeak_f1"],
                    "waveform_temporal_std": w_m["waveform_temporal_std"],
                    "waveform_amplitude_range": w_m["waveform_amplitude_range"],
                })

    summary_ablations = {}
    for abl_name in ablations:
        dicts = ablation_metrics[abl_name]
        summary_ablations[abl_name] = {
            k: float(np.mean([d[k] for d in dicts])) for k in dicts[0].keys()
        }

    # Latent-Dimension Perturbation Sensitivity Test (Section 7.2)
    print("[Diagnose Decoder] Running per-latent-dimension perturbation test...")
    latent_dim = post_latent.size(-1)
    emp_std_per_dim = post_latent.std(dim=(0, 1)).cpu().numpy()
    sensitivity_rows = []

    with torch.no_grad():
        for d in range(latent_dim):
            eps = 0.1 * max(float(emp_std_per_dim[d]), 1e-3)
            z_plus = post_latent.clone()
            z_plus[..., d] += eps

            wf_pert, _ = model.decoder(z_plus, c_summary, target_len=t_target)
            mean_abs_change = float((wf_pert - wf1).abs().mean().item())
            r_pert = compute_rhythm_debug_metrics(wf_pert, f_wf)
            r_base = compute_rhythm_debug_metrics(wf1, f_wf)
            timing_change_ms = float(abs(r_pert["next_rpeak_timing_mae_ms"] - r_base["next_rpeak_timing_mae_ms"]))

            sensitivity_rows.append({
                "latent_dimension": d,
                "empirical_std": float(emp_std_per_dim[d]),
                "perturbation_epsilon": eps,
                "mean_absolute_waveform_change": mean_abs_change,
                "next_rpeak_timing_change_ms": timing_change_ms,
            })

    # Save sensitivity CSV
    os.makedirs(args.output_dir, exist_ok=True)
    import pandas as pd
    pd.DataFrame(sensitivity_rows).to_csv(os.path.join(args.output_dir, "latent_dimension_sensitivity.csv"), index=False)

    # Gate 4 Interpretation Rules (Section 7.3)
    base_pearson = summary_ablations["posterior_latent_plus_correct_context"]["macro_pearson"]
    zero_pearson = summary_ablations["zero_latent_plus_correct_context"]["macro_pearson"]
    time_shuf_pearson = summary_ablations["time_shuffled_latent_plus_correct_context"]["macro_pearson"]
    batch_shuf_pearson = summary_ablations["batch_shuffled_latent_plus_correct_context"]["macro_pearson"]

    zero_drop = base_pearson - zero_pearson
    time_shuf_drop = base_pearson - time_shuf_pearson
    batch_shuf_drop = base_pearson - batch_shuf_pearson

    if zero_drop >= 0.20 and time_shuf_drop >= 0.15 and batch_shuf_drop >= 0.15:
        classification = "latent used strongly"
    elif zero_drop >= 0.10 or time_shuf_drop >= 0.08:
        classification = "latent used partially"
    else:
        classification = "latent mostly ignored"

    interpretation = [
        f"Classification: {classification}",
        f"Zero latent Pearson drop: {zero_drop:.4f}",
        f"Time shuffle Pearson drop: {time_shuf_drop:.4f}",
        f"Batch shuffle Pearson drop: {batch_shuf_drop:.4f}",
    ]

    gate_4_passed = bool(classification in ["latent used strongly", "latent used partially"])

    summary_data = {
        "checkpoint": args.checkpoint,
        "classification": classification,
        "zero_latent_pearson_drop": zero_drop,
        "time_shuffle_pearson_drop": time_shuf_drop,
        "batch_shuffle_pearson_drop": batch_shuf_drop,
        "ablations": summary_ablations,
        "interpretation": interpretation,
        "gate_4_passed": gate_4_passed,
    }

    save_debug_artifacts(
        output_dir=args.output_dir,
        summary_data=summary_data,
        config=cfg,
    )

    print(f"[Diagnose Decoder] Complete. Gate 4 Passed: {gate_4_passed} ({classification}). Artifacts saved to {args.output_dir}")
    print("Diagnosis Interpretation:", interpretation)


if __name__ == "__main__":
    main()
