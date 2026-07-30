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
from ecg_forecast.debug.checkpoint_loader import load_forecaster_checkpoint
from ecg_forecast.debug.reporting import save_debug_artifacts




class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def extract_features_and_targets(
    encoder: ContextEncoder,
    dataloader: DataLoader,
    device: str = "cpu",
    max_batches: int = 20,
):
    encoder.eval()

    feats = {
        "global": [],
        "boundary": [],
        "recent": [],
        "dynamic": [],
        "global_boundary": [],
        "global_recent": [],
        "boundary_recent": [],
        "global_boundary_recent": [],
    }

    targets_next_rpeak_ms = []

    with torch.no_grad():
        for batch_cnt, batch in enumerate(dataloader):
            if batch_cnt >= max_batches:
                break

            c_wf = batch["context_waveform"].to(device)
            f_wf = batch["future_waveform"].to(device)
            b, t_c, c = c_wf.shape

            ef = encoder.encode_features(c_wf)

            g = ef["global"]
            b_tok = ef["boundary"]
            r = ef["recent"]
            dyn = ef["dynamic"]

            g_b = torch.cat([g, b_tok], dim=-1)
            g_r = torch.cat([g, r], dim=-1)
            b_r = torch.cat([b_tok, r], dim=-1)
            g_b_r = torch.cat([g, b_tok, r], dim=-1)

            for i in range(b):
                c_np = c_wf[i, :, 1 if c >= 2 else 0].cpu().numpy()
                f_np = f_wf[i, :, 1 if c >= 2 else 0].cpu().numpy()

                f_peaks = detect_r_peaks(f_np, fs=100)
                next_r_ms = (f_peaks[0] / 100.0) * 1000.0 if len(f_peaks) > 0 else 1000.0

                feats["global"].append(g[i].cpu())
                feats["boundary"].append(b_tok[i].cpu())
                feats["recent"].append(r[i].cpu())
                feats["dynamic"].append(dyn[i].cpu())
                feats["global_boundary"].append(g_b[i].cpu())
                feats["global_recent"].append(g_r[i].cpu())
                feats["boundary_recent"].append(b_r[i].cpu())
                feats["global_boundary_recent"].append(g_b_r[i].cpu())

                targets_next_rpeak_ms.append(next_r_ms)

    out_feats = {k: torch.stack(v) for k, v in feats.items()}
    out_targets = torch.tensor(targets_next_rpeak_ms, dtype=torch.float32).unsqueeze(1)
    return out_feats, out_targets


def train_and_eval_probe_pair(
    X_train: torch.Tensor, Y_train: torch.Tensor,
    X_val: torch.Tensor, Y_val: torch.Tensor,
    probe_cls: type, epochs: int = 50
) -> float:
    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    probe = probe_cls(in_dim=X_train.size(1), out_dim=1)
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
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/incart")
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Dataset directory '{args.data_dir}' does not exist. Cannot run context phase probes.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Probe Context] Loading trained checkpoint: {args.checkpoint}")
    model, cfg = load_forecaster_checkpoint(args.checkpoint, device=device)
    encoder = model.context_encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    cfg.data.data_dir = args.data_dir
    checkpoint_name = os.path.basename(args.checkpoint).replace(".pt", "")
    if args.output_dir is None:
        args.output_dir = os.path.join("artifacts/debug/probe_context_phase", checkpoint_name)

    print(f"[Probe Context] Loading train and val datasets from {args.data_dir}...")
    train_loader, val_loader, _, _ = get_incart_dataloaders(config=cfg.data, batch_size=16, num_workers=0)

    print("[Probe Context] Extracting features from train and val split records...")
    train_feats, train_targets = extract_features_and_targets(encoder, train_loader, device=device, max_batches=args.num_batches)
    val_feats, val_targets = extract_features_and_targets(encoder, val_loader, device=device, max_batches=args.num_batches)

    probe_results = {}
    print("[Probe Context] Training Linear and MLP probes for next R-peak timing MAE (ms)...")
    for feat_key in train_feats.keys():
        X_tr, Y_tr = train_feats[feat_key], train_targets
        X_va, Y_va = val_feats[feat_key], val_targets

        linear_mae = train_and_eval_probe_pair(X_tr, Y_tr, X_va, Y_va, LinearProbe)
        mlp_mae = train_and_eval_probe_pair(X_tr, Y_tr, X_va, Y_va, MLPProbe)

        probe_results[feat_key] = {
            "linear_mae_ms": linear_mae,
            "mlp_mae_ms": mlp_mae,
            "best_mae_ms": min(linear_mae, mlp_mae),
        }
        print(f"  Feature '{feat_key}': Linear MAE = {linear_mae:.2f} ms | MLP MAE = {mlp_mae:.2f} ms")

    global_mae = probe_results["global"]["best_mae_ms"]
    best_comb_mae = min([v["best_mae_ms"] for k, v in probe_results.items() if k != "global"])

    rel_improvement = ((global_mae - best_comb_mae) / global_mae) * 100.0 if global_mae > 0 else 0.0
    gate_5_passed = bool(rel_improvement >= 20.0 or best_comb_mae <= 100.0)

    interpretation = [
        f"Global summary best MAE: {global_mae:.2f} ms",
        f"Best boundary-aware feature MAE: {best_comb_mae:.2f} ms",
        f"Relative improvement: {rel_improvement:.2f}%",
    ]

    summary_data = {
        "checkpoint": args.checkpoint,
        "feature_probe_results": probe_results,
        "global_mae_ms": global_mae,
        "best_combined_mae_ms": best_comb_mae,
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
