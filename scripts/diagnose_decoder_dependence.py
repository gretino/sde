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
    latent_grads = []
    context_grads = []

    print("[Diagnose Decoder] Running decoder input ablations and gradient sensitivity...")

    for batch_cnt, batch in enumerate(val_loader):
        if batch_cnt >= args.num_batches:
            break

        c_wf = batch["context_waveform"].to(device)
        f_wf = batch["future_waveform"].to(device)
        b = c_wf.size(0)

        # 1. Obtain posterior and prior latents and context summary
        c_summary, _, prior_mean, _ = model.context_encoder(c_wf)
        full_wf = torch.cat([c_wf, f_wf], dim=1)
        _, rec_path, post_mean, _ = model.posterior_encoder(full_wf, c_summary)

        ts = torch.linspace(0.04, 2.0, 50, device=device)
        with torch.no_grad():
            post_latent, _ = model.sde.integrate(
                z0=post_mean, ts=ts, context_summary=c_summary, recognition_path=rec_path, mode="posterior"
            )
            prior_latent, _ = model.sde.integrate(
                z0=prior_mean, ts=ts, context_summary=c_summary, mode="prior"
            )

        # Gradient sensitivity calculation
        lat_req = post_latent.detach().clone().requires_grad_(True)
        ctx_req = c_summary.detach().clone().requires_grad_(True)

        wf_pred, _ = model.decoder(lat_req, ctx_req)
        loss_dummy = wf_pred.sum()
        loss_dummy.backward()

        l_grad_norm = float(lat_req.grad.abs().mean().item())
        c_grad_norm = float(ctx_req.grad.abs().mean().item())
        latent_grads.append(l_grad_norm)
        context_grads.append(c_grad_norm)

        # 2. Evaluate Ablations
        with torch.no_grad():
            # A1: posterior_latent + correct_context
            wf1, _ = model.decoder(post_latent, c_summary)

            # A2: posterior_latent + zero_context
            wf2, _ = model.decoder(post_latent, torch.zeros_like(c_summary))

            # A3: zero_latent + correct_context
            wf3, _ = model.decoder(torch.zeros_like(post_latent), c_summary)

            # A4: time_shuffled_posterior_latent + correct_context
            perm_t = torch.randperm(post_latent.size(1))
            wf4, _ = model.decoder(post_latent[:, perm_t, :], c_summary)

            # A5: batch_shuffled_posterior_latent + correct_context
            perm_b = torch.randperm(b)
            wf5, _ = model.decoder(post_latent[perm_b], c_summary)

            # A6: posterior_latent + batch_shuffled_context
            wf6, _ = model.decoder(post_latent, c_summary[perm_b])

            # A7: prior_latent + correct_context
            wf7, _ = model.decoder(prior_latent, c_summary)

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

    mean_latent_grad = float(np.mean(latent_grads))
    mean_context_grad = float(np.mean(context_grads))
    grad_ratio = mean_latent_grad / (mean_context_grad + 1e-8)

    summary_ablations = {}
    for abl_name in ablations:
        dicts = ablation_metrics[abl_name]
        summary_ablations[abl_name] = {
            k: float(np.mean([d[k] for d in dicts])) for k in dicts[0].keys()
        }

    # Interpretation
    baseline_mse = summary_ablations["posterior_latent_plus_correct_context"]["mse"]
    zero_lat_mse = summary_ablations["zero_latent_plus_correct_context"]["mse"]
    time_shuf_mse = summary_ablations["time_shuffled_latent_plus_correct_context"]["mse"]
    batch_shuf_mse = summary_ablations["batch_shuffled_latent_plus_correct_context"]["mse"]

    interpretation = []
    if zero_lat_mse <= baseline_mse * 1.1:
        interpretation.append("CRITICAL: decoder context shortcut detected (zero latent performs as well as true latent)")
    if time_shuf_mse <= baseline_mse * 1.1:
        interpretation.append("decoder ignores temporal ordering of latent trajectory")
    if batch_shuf_mse <= baseline_mse * 1.1:
        interpretation.append("decoder ignores latent identity (uses only context summary)")
    if grad_ratio < 0.05:
        interpretation.append("CRITICAL: latent gradient sensitivity is near zero")

    gate_4_passed = bool(
        zero_lat_mse > baseline_mse * 1.5 and
        time_shuf_mse > baseline_mse * 1.3 and
        grad_ratio >= 0.10
    )

    summary_data = {
        "checkpoint": args.checkpoint,
        "mean_latent_grad_sensitivity": mean_latent_grad,
        "mean_context_grad_sensitivity": mean_context_grad,
        "gradient_ratio_latent_to_context": grad_ratio,
        "ablations": summary_ablations,
        "interpretation": interpretation,
        "gate_4_passed": gate_4_passed,
    }

    save_debug_artifacts(
        output_dir=args.output_dir,
        summary_data=summary_data,
        config=cfg,
    )

    print(f"[Diagnose Decoder] Complete. Gate 4 Passed: {gate_4_passed}. Summary saved to {args.output_dir}/summary.json")
    print("Diagnosis Interpretation:", interpretation)


if __name__ == "__main__":
    main()
