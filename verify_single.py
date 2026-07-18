import os
import argparse
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from sde.encoder import PhysiologicalEncoder, get_interpolated_latent_trajectory
from sde.solver import ContinuousSolver, FBase
from sde.baseline import NeuroSDEBaseline, PhaseTolerantDecoder
from sde.incart_dataset import IncartDataset, get_incart_splits
from sde.weight_utils import load_pretrained_ecg_fm
from run_incart_experiment import set_seed, compute_pearson_correlation, round_dict_floats

def find_best_shift(pred, target, max_shift=100):
    best_shift = 0
    best_mse = float('inf')
    for shift in range(-max_shift, max_shift + 1):
        if shift < 0:
            p_slice = pred[-shift:]
            t_slice = target[:shift]
        elif shift > 0:
            p_slice = pred[:-shift]
            t_slice = target[shift:]
        else:
            p_slice = pred
            t_slice = target
        
        if len(p_slice) == 0:
            continue
        mse = np.mean((p_slice - t_slice) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_shift = shift
    return best_shift

def compute_pearson_correlation_np(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    centered_x = x - mean_x
    centered_y = y - mean_y
    cov = np.sum(centered_x * centered_y)
    var_x = np.sum(centered_x ** 2)
    var_y = np.sum(centered_y ** 2)
    if var_x == 0 or var_y == 0:
        return 0.0
    return float(cov / np.sqrt(var_x * var_y))

def main():
    load_dotenv()
    
    # 1. Pre-parse configuration file argument
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--config", type=str, default="config/incart_config.yaml", help="Path to config YAML file")
    temp_args, _ = temp_parser.parse_known_args()
    
    config_defaults = {}
    if os.path.exists(temp_args.config):
        try:
            with open(temp_args.config, "r") as f:
                config_defaults = yaml.safe_load(f)
            print(f"Loaded parameters from config: {temp_args.config}")
        except Exception as e:
            print(f"Warning: Failed to load config from {temp_args.config}: {e}")
            
    # 2. Main parser
    parser = argparse.ArgumentParser(description="Single Sample Verification and Plot Generation")
    parser.add_argument("--config", type=str, default="config/incart_config.yaml", help="Path to config YAML file")
    parser.add_argument("--db-dir", type=str, default="/home/qfbqt/8TB/datasets/physionet.org/files/incartdb/1.0.0", help="Path to database directory")
    parser.add_argument("--weight-path", type=str, default=None, help="Path to mimic_iv_ecg_finetuned.pt weight file")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/neurosde_final.pt", help="Path to SDE checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb-run-name", type=str, default="incart-epoch5", help="Run name used for output folder")
    parser.add_argument("--sample-idx", type=int, default=0, help="Index of the test set sample to evaluate")
    parser.add_argument("--decoder-hidden-dim", type=int, default=256, help="Intermediate hidden dimension of the decoder")
    parser.add_argument("--latent-dim", type=int, default=32, help="Dimensionality of the continuous latent space")
    
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Splits and Datasets
    _, test_recs = get_incart_splits(args.db_dir, seed=args.seed)
    test_dataset = IncartDataset(args.db_dir, test_recs, use_cache=True)
    
    # Instantiate Model
    latent_dim = args.latent_dim
    leads = 12
    conv_layers = [(256, 2, 2)] * 4
    
    encoder = PhysiologicalEncoder(in_leads=leads, conv_layers=conv_layers, latent_dim=latent_dim)
    f_base = FBase(latent_dim=latent_dim, hidden_dim=64)
    solver = ContinuousSolver(f_base=f_base)
    decoder = PhaseTolerantDecoder(latent_dim=latent_dim, leads=leads, hidden_dim=args.decoder_hidden_dim)
    model = NeuroSDEBaseline(encoder, solver, decoder).to(device)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint not found at {args.checkpoint_path}!")
        print("Please train the model first.")
        return
        
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Fetch target sample
    context_wf, context_t, target_wf, target_t, _ = test_dataset[args.sample_idx]
    
    c_wf = context_wf.unsqueeze(0).to(device)
    c_t = context_t.to(device)
    t_wf = target_wf.unsqueeze(0).to(device)
    t_t = target_t.to(device)
    
    # Run predictions
    with torch.no_grad():
        # 1. Decoder-only context reconstruction
        z_context = get_interpolated_latent_trajectory(model.encoder, c_wf, c_t)
        recon_context = model.decoder(z_context, c_t)
        
        # 2. Oracle future reconstruction
        z_future_oracle = get_interpolated_latent_trajectory(model.encoder, t_wf, t_t)
        recon_oracle = model.decoder(z_future_oracle, t_t)
        
        # 3. SDE Evolved Prediction
        recon_sde, _ = model(c_wf, c_t, t_t)
        
    # Convert outputs to CPU numpy arrays for Lead II (index 1)
    orig_context_np = context_wf[:, 1].cpu().numpy()
    recon_context_np = recon_context[0, :, 1].cpu().numpy()
    
    orig_target_np = target_wf[:, 1].cpu().numpy()
    recon_oracle_np = recon_oracle[0, :, 1].cpu().numpy()
    recon_sde_np = recon_sde[0, :, 1].cpu().numpy()
    
    # Find the best shift (max 100 samples = 1.0 second)
    best_shift = find_best_shift(recon_sde_np, orig_target_np, max_shift=100)
    shift_seconds = best_shift * 0.01
    
    # Construct shifted array
    recon_sde_shifted_np = np.zeros_like(recon_sde_np)
    if best_shift < 0:
        recon_sde_shifted_np[:best_shift] = recon_sde_np[-best_shift:]
        recon_sde_shifted_np[best_shift:] = recon_sde_np[-1]
    elif best_shift > 0:
        recon_sde_shifted_np[best_shift:] = recon_sde_np[:-best_shift]
        recon_sde_shifted_np[:best_shift] = recon_sde_np[0]
    else:
        recon_sde_shifted_np = recon_sde_np.copy()
        
    # Compute Pearson correlations
    corr_context = float(compute_pearson_correlation(recon_context, c_wf))
    corr_oracle = float(compute_pearson_correlation(recon_oracle, t_wf))
    corr_sde = float(compute_pearson_correlation(recon_sde, t_wf))
    corr_sde_shifted = compute_pearson_correlation_np(recon_sde_shifted_np, orig_target_np)
    
    # Compute MSEs
    mse_context = float(torch.nn.MSELoss()(recon_context, c_wf).item())
    mse_oracle = float(torch.nn.MSELoss()(recon_oracle, t_wf).item())
    mse_sde = float(torch.nn.MSELoss()(recon_sde, t_wf).item())
    mse_sde_shifted = float(np.mean((recon_sde_shifted_np - orig_target_np) ** 2))
    
    metrics = {
        "sample_index": args.sample_idx,
        "context_reconstruction": {
            "pearson_correlation": corr_context,
            "mse": mse_context
        },
        "oracle_future_reconstruction": {
            "pearson_correlation": corr_oracle,
            "mse": mse_oracle
        },
        "sde_evolved_forecasting": {
            "pearson_correlation": corr_sde,
            "mse": mse_sde,
            "optimal_shift_seconds": shift_seconds,
            "aligned_pearson_correlation": corr_sde_shifted,
            "aligned_mse": mse_sde_shifted
        }
    }
    rounded_metrics = round_dict_floats(metrics)
    
    # Setup output paths
    run_name = args.wandb_run_name
    output_dir = os.path.join("output", run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_file = os.path.join(output_dir, "single_sample_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(rounded_metrics, f, indent=4)
        
    # Generate ECG Comparison Plot (3-panel)
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharey=True)
    
    # Panel 1: Context Window (Past 10 seconds)
    axes[0].plot(context_t.cpu().numpy(), orig_context_np, label="Original Context ECG", color="black", alpha=0.7)
    axes[0].plot(context_t.cpu().numpy(), recon_context_np, label="Decoder Reconstruction", color="blue", linestyle="--", alpha=0.9)
    axes[0].set_title(f"Context Window (Past 10s) - Lead II - (Pearson r: {rounded_metrics['context_reconstruction']['pearson_correlation']:.4f})")
    axes[0].set_ylabel("Normalized Voltage")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right")
    
    # Panel 2: Future Window Panel - Raw (Future 10 seconds)
    axes[1].plot(target_t.cpu().numpy(), orig_target_np, label="Original Target ECG", color="black", alpha=0.7)
    axes[1].plot(target_t.cpu().numpy(), recon_oracle_np, label="Oracle Decoder Reconstruction", color="green", linestyle=":", alpha=0.9)
    axes[1].plot(target_t.cpu().numpy(), recon_sde_np, label="Raw SDE Evolved Forecast", color="red", linestyle="--", alpha=0.9)
    axes[1].set_title(f"Future Window (Next 10s) - Lead II - Raw Forecast\n(Oracle r: {rounded_metrics['oracle_future_reconstruction']['pearson_correlation']:.4f} | SDE Forecast r: {rounded_metrics['sde_evolved_forecasting']['pearson_correlation']:.4f})")
    axes[1].set_ylabel("Normalized Voltage")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right")
    
    # Panel 3: Future Window Panel - Phase-Aligned (Future 10 seconds)
    axes[2].plot(target_t.cpu().numpy(), orig_target_np, label="Original Target ECG", color="black", alpha=0.7)
    axes[2].plot(target_t.cpu().numpy(), recon_sde_shifted_np, label=f"Phase-Aligned SDE Forecast (Shift: {rounded_metrics['sde_evolved_forecasting']['optimal_shift_seconds']:.2f}s)", color="purple", linestyle="--", alpha=0.9)
    axes[2].set_title(f"Future Window (Next 10s) - Lead II - Phase-Aligned Forecast (Aligned r: {rounded_metrics['sde_evolved_forecasting']['aligned_pearson_correlation']:.4f})")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("Normalized Voltage")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(loc="upper right")
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, "single_sample_comparison.png")
    plt.savefig(plot_file, dpi=150)
    plt.close()
    
    print("\n--- Single Sample Evaluation Summary ---")
    print(f"Sample Index: {args.sample_idx}")
    print(f"Context Recon     - MSE: {rounded_metrics['context_reconstruction']['mse']:.4f} | Pearson r: {rounded_metrics['context_reconstruction']['pearson_correlation']:.4f}")
    print(f"Oracle Recon      - MSE: {rounded_metrics['oracle_future_reconstruction']['mse']:.4f} | Pearson r: {rounded_metrics['oracle_future_reconstruction']['pearson_correlation']:.4f}")
    print(f"SDE Forecast      - MSE: {rounded_metrics['sde_evolved_forecasting']['mse']:.4f} | Pearson r: {rounded_metrics['sde_evolved_forecasting']['pearson_correlation']:.4f}")
    print(f"SDE Forecast (Al) - MSE: {rounded_metrics['sde_evolved_forecasting']['aligned_mse']:.4f} | Pearson r: {rounded_metrics['sde_evolved_forecasting']['aligned_pearson_correlation']:.4f} | Shift: {rounded_metrics['sde_evolved_forecasting']['optimal_shift_seconds']:.2f}s")
    print("----------------------------------------")
    print(f"3-panel single sample comparison plot saved to: {plot_file}")
    print(f"Single sample metrics saved to:                  {metrics_file}")

if __name__ == "__main__":
    main()
