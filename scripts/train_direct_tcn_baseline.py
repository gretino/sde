#!/usr/bin/env python3
"""Script 19: train_direct_tcn_baseline.py - Direct 12-Lead Forecasting TCN Baseline (Gate 1)."""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


from ecg_forecast.config import Config
from ecg_forecast.data.incart import get_incart_dataloaders
from ecg_forecast.losses.morphology import compute_morphology_loss
from ecg_forecast.debug.metrics import (
    compute_waveform_debug_metrics,
    compute_rhythm_debug_metrics,
)
from ecg_forecast.debug.reporting import save_debug_artifacts


class PhasePreservingTCNForecaster(nn.Module):
    """Phase-preserving Temporal Convolutional Network mapping context directly to future ECG."""
    def __init__(self, num_leads: int = 12, future_samples: int = 50, hidden_dim: int = 128):
        super().__init__()
        self.future_samples = future_samples
        self.num_leads = num_leads

        # Causal Residual 1D Encoder (preserve temporal dimension without global avg pool)
        self.conv1 = nn.Conv1d(num_leads, 64, kernel_size=7, stride=2, padding=3)  # 500 -> 250
        self.conv2 = nn.Conv1d(64, hidden_dim, kernel_size=5, stride=2, padding=2)  # 250 -> 125
        self.res = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )

        # Attention pooling for global summary
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Dynamic context projection from [global, boundary, recent]
        self.dynamic_proj = nn.Linear(hidden_dim * 3, hidden_dim)

        # Temporal future decoder
        self.up = nn.Upsample(size=future_samples, mode="linear", align_corners=False)
        self.decoder_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, num_leads, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 500, 12] -> transpose to [B, 12, 500]
        b = x.size(0)
        x_t = x.transpose(1, 2)
        h = F.gelu(self.conv1(x_t))
        h = F.gelu(self.conv2(h))
        h = F.gelu(h + self.res(h))  # [B, hidden_dim, 125]

        tokens = h.transpose(1, 2)  # [B, 125, hidden_dim]
        weights = F.softmax(self.attn(tokens), dim=1)
        global_summary = (tokens * weights).sum(dim=1)
        boundary_token = tokens[:, -1, :]
        recent_summary = tokens[:, -25:, :].mean(dim=1)

        concat_feats = torch.cat([global_summary, boundary_token, recent_summary], dim=-1)
        c_dynamic = F.gelu(self.dynamic_proj(concat_feats))  # [B, hidden_dim]

        # Expand temporal grid and decode
        c_grid = c_dynamic.unsqueeze(-1)  # [B, hidden_dim, 1]
        c_up = self.up(c_grid)             # [B, hidden_dim, future_samples]
        out_t = self.decoder_conv(c_up)    # [B, num_leads, future_samples]
        return out_t.transpose(1, 2)        # [B, future_samples, num_leads]


class BeatRepeatBaseline:
    """Non-parametric baseline that repeats the final context cardiac cycle into the future."""
    def predict(self, context_waveform: torch.Tensor, future_samples: int = 50) -> torch.Tensor:
        # Repeat final 1.0s (100 samples) or required future_samples from recent context
        rec = context_waveform[:, -100:, :]
        repeats = (future_samples // rec.size(1)) + 1
        rep = rec.repeat(1, repeats, 1)
        return rep[:, :future_samples, :]


def main():
    parser = argparse.ArgumentParser(description="Train direct 12-lead TCN forecasting baseline (Gate 1).")
    parser.add_argument("--data_dir", type=str, default="data/incart")
    parser.add_argument("--horizon_sec", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--overfit_windows", type=int, default=0, help="If > 0, overfit on tiny subset of N windows")
    parser.add_argument("--output_dir", type=str, default="artifacts/debug/train_direct_tcn_baseline/incart")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Dataset directory '{args.data_dir}' does not exist. Cannot train direct TCN baseline.")

    cfg = Config()
    cfg.data.data_dir = args.data_dir
    cfg.data.future_seconds = args.horizon_sec

    print(f"[Direct TCN Baseline] Loading dataloaders for {args.horizon_sec}s forecast from {args.data_dir}...")
    train_loader, val_loader, _, _ = get_incart_dataloaders(config=cfg.data, batch_size=args.batch_size, num_workers=0)

    if args.overfit_windows > 0:
        from torch.utils.data import Subset, DataLoader
        from ecg_forecast.data.collate import ecg_collate_fn
        tiny_sub = Subset(train_loader.dataset, range(min(args.overfit_windows, len(train_loader.dataset))))
        train_loader = DataLoader(tiny_sub, batch_size=min(args.batch_size, args.overfit_windows), shuffle=False, collate_fn=ecg_collate_fn)
        val_loader = train_loader

    future_samples = int(round(args.horizon_sec * 100))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PhasePreservingTCNForecaster(num_leads=cfg.model.num_leads, future_samples=future_samples).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    beat_baseline = BeatRepeatBaseline()

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

        # Validation & Beat Repeat Baseline
        model.eval()
        val_preds = []
        val_targets = []
        beat_preds = []
        with torch.no_grad():
            for batch in val_loader:
                c_wf = batch["context_waveform"].to(device)
                f_wf = batch["future_waveform"][:, :future_samples, :].to(device)

                pred = model(c_wf)
                b_pred = beat_baseline.predict(c_wf, future_samples=future_samples)

                val_preds.append(pred.cpu())
                val_targets.append(f_wf.cpu())
                beat_preds.append(b_pred.cpu())

        all_val_preds = torch.cat(val_preds, dim=0)
        all_val_targets = torch.cat(val_targets, dim=0)
        all_beat_preds = torch.cat(beat_preds, dim=0)

        w_m = compute_waveform_debug_metrics(all_val_preds, all_val_targets)
        r_m = compute_rhythm_debug_metrics(all_val_preds, all_val_targets)
        beat_w_m = compute_waveform_debug_metrics(all_beat_preds, all_val_targets)

        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            print(
                f"  Epoch {epoch+1:02d} | Train Loss: {np.mean(train_losses):.4f} | "
                f"Val MSE: {w_m['mse']:.4f} | Val Pearson: {w_m['macro_pearson']:.4f} | "
                f"Beat-Repeat Pearson: {beat_w_m['macro_pearson']:.4f} | "
                f"Val R-peak F1: {r_m['rpeak_f1']:.4f}"
            )

    gate_1_passed = bool(
        w_m["macro_pearson"] >= 0.70 and
        r_m["rpeak_f1"] >= 0.70 and
        r_m["zero_rpeak_forecast_pct"] < 20.0
    )

    summary_data = {
        "horizon_sec": args.horizon_sec,
        "epochs": args.epochs,
        "overfit_mode": args.overfit_windows > 0,
        "validation_waveform": w_m,
        "beat_repeat_waveform": beat_w_m,
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
