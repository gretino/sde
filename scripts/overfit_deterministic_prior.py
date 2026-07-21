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


def main():
    parser = argparse.ArgumentParser(description="Overfit deterministic prior on tiny 32-window dataset (Gate 2).")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional Stage A checkpoint path (Version A)")
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

    print(f"[Overfit Deterministic Prior] Loading data from {args.data_dir} for 32-window overfit test...")
    train_loader, _, _, _ = get_incart_dataloaders(config=cfg.data, batch_size=args.batch_size, num_workers=0)

    from ecg_forecast.data.collate import ecg_collate_fn

    # Extract exactly 32 samples
    tiny_dataset = Subset(train_loader.dataset, range(min(32, len(train_loader.dataset))))
    tiny_loader = DataLoader(tiny_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ecg_collate_fn)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        print(f"[Overfit Deterministic Prior] Loading Stage A checkpoint from {args.checkpoint} (Version A)...")
        from ecg_forecast.debug.checkpoint_loader import load_forecaster_checkpoint
        model, cfg = load_forecaster_checkpoint(args.checkpoint, device=device)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.context_encoder.fc_mean.parameters():
            p.requires_grad = True
        for p in model.sde.sde_func.prior_drift_net.parameters():
            p.requires_grad = True
    else:
        print("[Overfit Deterministic Prior] Training from scratch (Version B: unfreezing context encoder, prior drift, and decoder)...")
        model = LatentSDEForecaster(config=cfg.model).to(device)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.context_encoder.parameters():
            p.requires_grad = True
        for p in model.sde.sde_func.prior_drift_net.parameters():
            p.requires_grad = True
        for p in model.decoder.parameters():
            p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)


    t_samples = int(round(args.horizon_sec * 100))
    t_latent = int(round(args.horizon_sec * 25))
    ts = torch.linspace(0.04, args.horizon_sec, t_latent, device=device)

    history = []
    print(f"[Overfit Deterministic Prior] Training for {args.epochs} epochs on {len(tiny_dataset)} windows...")

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

            # Force zero diffusion
            with torch.amp.autocast("cuda", enabled=False):
                latent_path, _ = model.sde.integrate(
                    z0=z0_p, ts=ts, context_summary=c_summary, mode="prior", brownian_motion=None
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
        "dataset": "32_window_subset",
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

    print(f"[Overfit Deterministic Prior] Complete. Gate 2 Passed: {gate_2_passed}. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
