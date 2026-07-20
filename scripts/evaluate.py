import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from ecg_forecast.config import load_config
from ecg_forecast.data import ECGWindowDataset, ecg_collate_fn
from ecg_forecast.models import LatentSDEForecaster
from ecg_forecast.training import load_checkpoint
from ecg_forecast.metrics import compute_waveform_metrics, compute_rhythm_metrics, compute_uncertainty_metrics


def evaluate_repeat_baseline(test_loader: DataLoader) -> dict:
    """Evaluates Repeat-Context Baseline on dataset (repeating last 2s of context)."""
    results = {"mse": [], "mae": [], "pearson": [], "rpeak_f1": []}

    for batch in test_loader:
        c_wf = batch["context_waveform"]  # [B, 500, num_leads]
        f_wf = batch["future_waveform"]   # [B, 200, num_leads]

        # Repeat last 200 samples of context window
        repeat_pred = c_wf[:, -200:, :]

        wf_m = compute_waveform_metrics(repeat_pred, f_wf)
        rhythm_m = compute_rhythm_metrics(repeat_pred, batch["future_r_peaks"])

        results["mse"].append(wf_m["mse"])
        results["mae"].append(wf_m["mae"])
        results["pearson"].append(wf_m["pearson"])
        results["rpeak_f1"].append(rhythm_m["rpeak_f1"])

    return {k: float(np.mean(v)) for k, v in results.items()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Latent SDE Forecaster")
    parser.add_argument("--config", type=str, default="configs/incart_12lead.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/incart_12lead/final_best.pt")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=16)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = ECGWindowDataset(config=cfg.data, split=args.split)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=ecg_collate_fn,
    )

    print(f"--- Evaluating Repeat-Context Baseline on {args.split} split ---")
    base_m = evaluate_repeat_baseline(test_loader)
    for k, v in base_m.items():
        print(f"  Baseline {k}: {v:.4f}")

    if os.path.exists(args.checkpoint):
        print(f"\n--- Evaluating Latent SDE Model ({args.checkpoint}) ---")
        model = LatentSDEForecaster(config=cfg.model)
        load_checkpoint(args.checkpoint, model=model, device=str(device))
        model.to(device)
        model.eval()

        model_results = {"mse": [], "mae": [], "pearson": [], "rpeak_f1": [], "coverage_90": [], "width_90": []}

        with torch.no_grad():
            for batch in test_loader:
                c_wf = batch["context_waveform"].to(device)
                f_wf = batch["future_waveform"].to(device)

                # Prior forecast 16 samples
                forecast_out = model.forward_prior(c_wf, num_samples=args.num_samples)
                # forecast_out.waveform_mean shape: [B * num_samples, 200, num_leads]
                b = c_wf.size(0)
                n_s = args.num_samples
                num_leads = cfg.model.num_leads

                samples_reshaped = forecast_out.waveform_mean.view(b, n_s, 200, num_leads).transpose(0, 1)

                ensemble_mean = samples_reshaped.mean(dim=0)  # [B, 200, num_leads]

                wf_m = compute_waveform_metrics(ensemble_mean, f_wf)
                rhythm_m = compute_rhythm_metrics(ensemble_mean, batch["future_r_peaks"])
                uncert_m = compute_uncertainty_metrics(samples_reshaped, f_wf)

                model_results["mse"].append(wf_m["mse"])
                model_results["mae"].append(wf_m["mae"])
                model_results["pearson"].append(wf_m["pearson"])
                model_results["rpeak_f1"].append(rhythm_m["rpeak_f1"])
                model_results["coverage_90"].append(uncert_m["coverage_90"])
                model_results["width_90"].append(uncert_m["width_90"])

        avg_res = {k: float(np.mean(v)) for k, v in model_results.items()}
        print("\n--- Latent SDE Model Evaluation Results ---")
        for k, v in avg_res.items():
            print(f"  Model {k}: {v:.4f}")
    else:
        print(f"Checkpoint {args.checkpoint} not found. Skipping model evaluation.")


if __name__ == "__main__":
    main()
