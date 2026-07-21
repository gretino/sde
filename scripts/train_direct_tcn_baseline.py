#!/usr/bin/env python3
"""Script 19: train_direct_tcn_baseline.py - Direct 12-Lead Forecasting TCN Baseline (Gate 1)."""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ecg_forecast.config import Config
from ecg_forecast.data.incart import get_incart_dataloaders
from ecg_forecast.losses.morphology import compute_morphology_loss
from ecg_forecast.debug.metrics import (
    compute_waveform_debug_metrics,
    compute_rhythm_debug_metrics,
)
from ecg_forecast.debug.reporting import save_debug_artifacts


class DirectTCNForecaster(nn.Module):
    """Deterministic Temporal Convolutional Network mapping context [B, 500, 12] directly to future [B, T_fut, 12]."""
    def __init__(self, num_leads: int = 12, future_samples: int = 50, hidden_dim: int = 128):
        super().__init__()
        self.future_samples = future_samples

        self.encoder = nn.Sequential(
            nn.Conv1d(num_leads, hidden_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.future_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, future_samples * num_leads),
        )
        self.num_leads = num_leads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 500, 12] -> transpose to [B, 12, 500]
        b = x.size(0)
        x_t = x.transpose(1, 2)
        h = self.encoder(x_t).squeeze(-1)  # [B, hidden_dim]
        out_flat = self.future_head(h)     # [B, future_samples * 12]
        out = out_flat.view(b, self.future_samples, self.num_leads)
        return out


def main():
    parser = argparse.ArgumentParser(description="Train direct 12-lead TCN forecasting baseline (Gate 1).")
    parser.add_argument("--data_dir", type=str, default="data/incart")
    parser.add_argument("--horizon_sec", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="artifacts/debug/train_direct_tcn_baseline/incart")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Dataset directory '{args.data_dir}' does not exist. Cannot train direct TCN baseline.")

    cfg = Config()
    cfg.data.data_dir = args.data_dir
    cfg.data.future_seconds = args.horizon_sec

    print(f"[Direct TCN Baseline] Loading dataloaders for {args.horizon_sec}s forecast from {args.data_dir}...")
    train_loader, val_loader, _, _ = get_incart_dataloaders(config=cfg.data, batch_size=args.batch_size, num_workers=0)

    future_samples = int(round(args.horizon_sec * 100))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DirectTCNForecaster(num_leads=cfg.model.num_leads, future_samples=future_samples).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    print(f"[Direct TCN Baseline] Training model for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            c_wf = batch["context_waveform"].to(device)
            f_wf = batch["future_waveform"][:, :future_samples, :].to(device)

            optimizer.zero_grad()
            pred = model(c_wf)

            loss_l1 = torch.abs(pred - f_wf).mean()
            loss_morph, _ = compute_morphology_loss(pred, f_wf)
            total_loss = loss_l1 + loss_morph

            total_loss.backward()
            optimizer.step()
            train_losses.append(float(total_loss.item()))

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                c_wf = batch["context_waveform"].to(device)
                f_wf = batch["future_waveform"][:, :future_samples, :].to(device)

                pred = model(c_wf)
                val_preds.append(pred.cpu())
                val_targets.append(f_wf.cpu())

        all_val_preds = torch.cat(val_preds, dim=0)
        all_val_targets = torch.cat(val_targets, dim=0)

        w_m = compute_waveform_debug_metrics(all_val_preds, all_val_targets)
        r_m = compute_rhythm_debug_metrics(all_val_preds, all_val_targets)

        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            print(
                f"  Epoch {epoch+1:02d} | Train Loss: {np.mean(train_losses):.4f} | "
                f"Val MSE: {w_m['mse']:.4f} | Val Pearson: {w_m['macro_pearson']:.4f} | "
                f"Val R-peak F1: {r_m['rpeak_f1']:.4f} | Zero R-peak Pct: {r_m['zero_rpeak_forecast_pct']:.1f}%"
            )

    gate_1_passed = bool(
        w_m["macro_pearson"] >= 0.70 and
        r_m["rpeak_f1"] >= 0.70 and
        r_m["zero_rpeak_forecast_pct"] < 20.0
    )

    summary_data = {
        "horizon_sec": args.horizon_sec,
        "epochs": args.epochs,
        "validation_waveform": w_m,
        "validation_rhythm": r_m,
        "gate_1_passed": gate_1_passed,
    }

    save_debug_artifacts(
        output_dir=args.output_dir,
        summary_data=summary_data,
        config=cfg,
    )

    print(f"[Direct TCN Baseline] Complete. Gate 1 Passed: {gate_1_passed}. Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
