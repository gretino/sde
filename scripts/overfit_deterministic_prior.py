#!/usr/bin/env python3
"""Script 18: overfit_deterministic_prior.py - Deterministic Prior Overfitting Tool (Gate 2)."""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ecg_forecast.config import Config
from ecg_forecast.data.incart import get_incart_dataloaders
from ecg_forecast.models.latent_sde_forecaster import LatentSDEForecaster
from ecg_forecast.losses.morphology import compute_morphology_loss
from ecg_forecast.debug.metrics import (
    compute_waveform_debug_metrics,
    compute_rhythm_debug_metrics,
)
from ecg_forecast.debug.reporting import save_debug_artifacts


def select_distributed_tiny_windows(dataset, target_count: int = 32):
    """Selects 32 windows distributed across multiple records rather than 32 contiguous windows from record 0."""
    total_len = len(dataset)
    if total_len <= target_count:
        return list(range(total_len))

    # Stratified step sampling across dataset records
    indices = np.linspace(0, total_len - 1, target_count, dtype=int).tolist()
    return indices


def main():
    parser = argparse.ArgumentParser(description="Overfit deterministic prior on tiny 32-window dataset (Gate 2).")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional Stage A checkpoint path (Gate 2B)")
    parser.add_argument("--data_dir", type=str, default="data/incart")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--horizon_sec", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="artifacts/debug/overfit_deterministic_prior/incart")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Dataset directory '{args.data_dir}' does not exist. Cannot run deterministic prior overfit.")

    cfg = Config()
    cfg.data.data_dir = args.data_dir
    cfg.data.future_seconds = args.horizon_sec

    print(f"[Overfit Deterministic Prior] Loading data from {args.data_dir} for 32-window distributed overfit test...")
    train_loader, _, _, _ = get_incart_dataloaders(config=cfg.data, batch_size=args.batch_size, num_workers=0)

    from ecg_forecast.data.collate import ecg_collate_fn
    from ecg_forecast.utils.timegrid import make_latent_times

    # Distributed 32-window selection across multiple records
    dist_indices = select_distributed_tiny_windows(train_loader.dataset, target_count=32)
    tiny_dataset = Subset(train_loader.dataset, dist_indices)
    tiny_loader = DataLoader(tiny_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_gate_2b = args.checkpoint is not None and os.path.exists(args.checkpoint)

    if is_gate_2b:
        print(f"[Overfit Deterministic Prior] Gate 2B: Loading Stage A checkpoint from {args.checkpoint} (Frozen representation)...")
        from ecg_forecast.debug.checkpoint_loader import load_forecaster_checkpoint
        model, cfg = load_forecaster_checkpoint(args.checkpoint, device=device)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.context_encoder.fc_mean.parameters():
            p.requires_grad = True
        for p in model.sde.sde_func.prior_drift_net.parameters():
            p.requires_grad = True
        gate_mode = "Gate_2B_Frozen_Stage_A"
    else:
        print("[Overfit Deterministic Prior] Gate 2A: Full-system memorization (Training context encoder, prior drift, and decoder)...")
        model = LatentSDEForecaster(config=cfg.model).to(device)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.context_encoder.parameters():
            p.requires_grad = True
        for p in model.sde.sde_func.prior_drift_net.parameters():
            p.requires_grad = True
        for p in model.decoder.parameters():
            p.requires_grad = True
        gate_mode = "Gate_2A_Full_System"

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)

    t_samples = int(round(args.horizon_sec * 100))
    ts = make_latent_times(future_samples=t_samples, sampling_rate=100, latent_rate=25, device=device)

    history = []
    print(f"[Overfit Deterministic Prior] [{gate_mode}] Training for {args.epochs} epochs on {len(tiny_dataset)} distributed windows...")

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        epoch_preds = []
        epoch_targets = []
        epoch_latents = []

        for batch in tiny_loader:
            c_wf = batch["context_waveform"].to(device)
            f_wf = batch["future_waveform"][:, :t_samples, :].to(device)

            optimizer.zero_grad()

            c_summary, _, prior_mean, _ = model.context_encoder(c_wf)
            z0_p = prior_mean

            # Exact deterministic SDE mode (deterministic=True)
            with torch.amp.autocast("cuda", enabled=False):
                latent_path, _ = model.sde.integrate(
                    z0=z0_p, ts=ts, context_summary=c_summary, mode="prior", deterministic=True
                )
                wf_pred, _ = model.decoder(latent_path, c_summary, target_len=f_wf.size(1))

                loss_l1 = torch.abs(wf_pred - f_wf).mean()
                loss_morph, _ = compute_morphology_loss(wf_pred, f_wf)
                loss = loss_l1 + loss_morph

                loss.backward()
                optimizer.step()

            epoch_losses.append(float(loss.item()))
            epoch_preds.append(wf_pred.detach())
            epoch_targets.append(f_wf.detach())
            epoch_latents.append(latent_path.detach())

        all_preds = torch.cat(epoch_preds, dim=0)
        all_targets = torch.cat(epoch_targets, dim=0)
        all_latents = torch.cat(epoch_latents, dim=0)

        w_m = compute_waveform_debug_metrics(all_preds, all_targets)
        r_m = compute_rhythm_debug_metrics(all_preds, all_targets)
        lat_std = float(all_latents.std(dim=1).mean().item())

        epoch_record = {
            "epoch": epoch + 1,
            "loss": float(np.mean(epoch_losses)),
            "mse": w_m["mse"],
            "mae": w_m["mae"],
            "macro_pearson": w_m["macro_pearson"],
            "rpeak_f1": r_m["rpeak_f1"],
            "latent_temporal_std": lat_std,
        }
        history.append(epoch_record)

        if (epoch + 1) % 20 == 0 or epoch == args.epochs - 1:
            print(
                f"  Epoch {epoch+1:03d} | Loss: {epoch_record['loss']:.4f} | "
                f"MSE: {w_m['mse']:.4f} | Macro Pearson: {w_m['macro_pearson']:.4f} | "
                f"R-peak F1: {r_m['rpeak_f1']:.4f} | Latent Std: {lat_std:.4f}"
            )

    final_m = history[-1]
    gate_2_passed = bool(
        final_m["macro_pearson"] >= 0.90 and
        final_m["rpeak_f1"] >= 0.90 and
        final_m["latent_temporal_std"] > 1e-4
    )

    summary_data = {
        "gate_mode": gate_mode,
        "dataset": "32_distributed_windows",
        "horizon_sec": args.horizon_sec,
        "final_metrics": final_m,
        "history": history,
        "gate_2_passed": gate_2_passed,
    }

    save_debug_artifacts(
        output_dir=args.output_dir,
        summary_data=summary_data,
        config=cfg,
    )

    print(f"[Overfit Deterministic Prior] [{gate_mode}] Complete. Gate 2 Passed: {gate_2_passed}. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
