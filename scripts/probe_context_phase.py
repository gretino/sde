#!/usr/bin/env python3
"""Script 17: probe_context_phase.py - Context Boundary & Cardiac Phase Probing Tool (Gate 5)."""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ecg_forecast.config import Config
from ecg_forecast.data.incart import get_incart_dataloaders
from ecg_forecast.metrics.rhythm import detect_r_peaks
from ecg_forecast.models.context_encoder import ContextEncoder
from ecg_forecast.debug.reporting import save_debug_artifacts


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def extract_context_features_and_targets(
    encoder: ContextEncoder,
    dataloader: DataLoader,
    device: str = "cpu",
    max_batches: int = 10,
):
    encoder.eval()

    feats_global = []
    feats_final_token = []
    feats_recent_mean = []
    feats_combined = []

    targets_next_rpeak_ms = []
    targets_future_hr = []

    with torch.no_grad():
        for batch_cnt, batch in enumerate(dataloader):
            if batch_cnt >= max_batches:
                break

            c_wf = batch["context_waveform"].to(device)
            f_wf = batch["future_waveform"].to(device)
            b, t_c, c = c_wf.shape

            c_summary, c_tokens, _, _ = encoder(c_wf)

            # Feature extractions
            # c_summary: [B, C_dim]
            # c_tokens: [B, T_lat, D]
            final_token = c_tokens[:, -1, :]  # [B, D]
            recent_mean = c_tokens[:, -25:, :].mean(dim=1)  # [B, D]

            # Downsample global summary to match D if needed or project
            combined = torch.cat([c_summary, final_token, recent_mean], dim=-1)

            for i in range(b):
                c_np = c_wf[i, :, 1 if c >= 2 else 0].cpu().numpy()
                f_np = f_wf[i, :, 1 if c >= 2 else 0].cpu().numpy()

                c_peaks = detect_r_peaks(c_np, fs=100)
                f_peaks = detect_r_peaks(f_np, fs=100)

                # Next R-peak timing from boundary (t=0) in ms
                if len(f_peaks) > 0:
                    next_r_ms = (f_peaks[0] / 100.0) * 1000.0
                else:
                    next_r_ms = 1000.0  # Fallback target if no peaks

                # Future HR
                f_hr = (len(f_peaks) / 2.0) * 60.0

                feats_global.append(c_summary[i].cpu())
                feats_final_token.append(final_token[i].cpu())
                feats_recent_mean.append(recent_mean[i].cpu())
                feats_combined.append(combined[i].cpu())

                targets_next_rpeak_ms.append(next_r_ms)
                targets_future_hr.append(f_hr)

    return {
        "global": torch.stack(feats_global),
        "final_token": torch.stack(feats_final_token),
        "recent_mean": torch.stack(feats_recent_mean),
        "combined": torch.stack(feats_combined),
    }, torch.tensor(targets_next_rpeak_ms, dtype=torch.float32).unsqueeze(1)


def train_and_eval_probe(X: torch.Tensor, Y: torch.Tensor, epochs: int = 50) -> float:
    n = X.size(0)
    train_n = int(n * 0.8)
    X_train, X_val = X[:train_n], X[train_n:]
    Y_train, Y_val = Y[:train_n], Y[train_n:]

    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    probe = LinearProbe(in_dim=X.size(1), out_dim=1)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    probe.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            out = probe(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        val_pred = probe(X_val)
        mae = float(criterion(val_pred, Y_val).item())
    return mae


def main():
    parser = argparse.ArgumentParser(description="Probe context representation for cardiac boundary phase (Gate 5).")
    parser.add_argument("--data_dir", type=str, default="data/incart")
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="artifacts/debug/probe_context_phase/incart")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Dataset directory '{args.data_dir}' does not exist. Cannot run context phase probes.")

    cfg = Config()
    cfg.data.data_dir = args.data_dir

    print(f"[Probe Context] Loading data from {args.data_dir}...")
    _, val_loader, _, _ = get_incart_dataloaders(config=cfg.data, batch_size=16, num_workers=0)

    encoder = ContextEncoder(num_leads=cfg.model.num_leads, context_dim=cfg.model.context_dim, latent_dim=cfg.model.latent_dim)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder.to(device)

    print("[Probe Context] Extracting context representations and timing targets...")
    feats_dict, Y_next_rpeak = extract_context_features_and_targets(encoder, val_loader, device=device, max_batches=args.num_batches)

    results = {}
    print("[Probe Context] Training linear probes for next R-peak timing MAE (ms)...")
    for feat_name, X_feat in feats_dict.items():
        mae = train_and_eval_probe(X_feat, Y_next_rpeak)
        results[feat_name] = {"next_rpeak_timing_mae_ms": mae}
        print(f"  Feature '{feat_name}': Next R-peak Timing MAE = {mae:.2f} ms")

    global_mae = results["global"]["next_rpeak_timing_mae_ms"]
    combined_mae = results["combined"]["next_rpeak_timing_mae_ms"]

    rel_improvement = ((global_mae - combined_mae) / global_mae) * 100.0 if global_mae > 0 else 0.0
    gate_5_passed = bool(rel_improvement >= 20.0 or combined_mae < 100.0)

    interpretation = []
    if rel_improvement >= 20.0:
        interpretation.append(f"Combined boundary representation improves timing MAE by {rel_improvement:.1f}% over global summary alone.")
    else:
        interpretation.append("Boundary token provides marginal improvement over global summary; consider architecture enhancement.")

    summary_data = {
        "feature_probe_results": results,
        "relative_improvement_pct": rel_improvement,
        "gate_5_passed": gate_5_passed,
        "interpretation": interpretation,
    }

    save_debug_artifacts(
        output_dir=args.output_dir,
        summary_data=summary_data,
        config=cfg,
    )

    print(f"[Probe Context] Complete. Gate 5 Passed: {gate_5_passed}. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
