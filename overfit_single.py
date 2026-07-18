import os
import argparse
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import torchcde

from sde.encoder import PhysiologicalEncoder, get_interpolated_latent_trajectory
from sde.baseline import PhaseTolerantDecoder
from sde.incart_dataset import IncartDataset, get_incart_splits
from sde.loss import PhaseTolerantWaveformLoss
from run_incart_experiment import set_seed, compute_pearson_correlation, round_dict_floats

def save_overfit_plot(step, times, original, reconstructed, output_path, pearson_r=None):
    plt.figure(figsize=(12, 5))
    plt.plot(times, original, label="Original ECG (Lead II)", color="black", alpha=0.7)
    plt.plot(times, reconstructed, label=f"Step {step} Reconstruction", color="red", linestyle="--", alpha=0.9)
    title = f"Overfitting Step {step}"
    if pearson_r is not None:
        title += f" (Pearson r: {pearson_r:.4f})"
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Normalized Voltage")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.savefig(output_path, dpi=150)
    plt.close()

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
    parser = argparse.ArgumentParser(description="Overfit one single sample sanity check")
    parser.add_argument("--config", type=str, default="config/incart_config.yaml", help="Path to config YAML file")
    parser.add_argument("--db-dir", type=str, default="/home/qfbqt/8TB/datasets/physionet.org/files/incartdb/1.0.0", help="Path to database directory")
    parser.add_argument("--weight-path", type=str, default=None, help="Path to mimic_iv_ecg_finetuned.pt weight file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb-run-name", type=str, default="incart-epoch5-fix1", help="Run name used for output folder")
    parser.add_argument("--sample-idx", type=int, default=0, help="Index of the test set sample to overfit")
    parser.add_argument("--latent-dim", type=int, default=64, help="Dimensionality of latent space")
    parser.add_argument("--decoder-hidden-dim", type=int, default=512, help="Hidden dimension of decoder")
    parser.add_argument("--epochs", type=int, default=500, help="Number of overfitting iterations")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for overfitting")
    
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Splits and Datasets
    train_recs, _ = get_incart_splits(args.db_dir, seed=args.seed)
    dataset = IncartDataset(args.db_dir, train_recs, use_cache=True)
    
    # Fetch a single context waveform
    context_wf, context_t, _, _, _ = dataset[args.sample_idx]
    
    # Context waveform shape is [1000, 12]
    # Expand to batch size 1 for training: [1, 1000, 12]
    c_wf = context_wf.unsqueeze(0).to(device)
    c_t = context_t.to(device)
    
    # Instantiate Encoder & Decoder only
    leads = 12
    conv_layers = [(256, 2, 2)] * 4
    encoder = PhysiologicalEncoder(in_leads=leads, conv_layers=conv_layers, latent_dim=args.latent_dim).to(device)
    decoder = PhaseTolerantDecoder(latent_dim=args.latent_dim, leads=leads, hidden_dim=args.decoder_hidden_dim).to(device)
    
    # Optimization: Unfreeze the encoder but use a 100x smaller learning rate for CDE params
    encoder.train()
    for param in encoder.parameters():
        param.requires_grad = True
        
    with torch.no_grad():
        patched_features = encoder.patcher(c_wf)
        batch_size, T_prime, _ = patched_features.shape
        new_times = torch.linspace(c_t[0].item(), c_t[-1].item(), T_prime, device=device)
        time_channel = new_times.unsqueeze(0).unsqueeze(-1).expand(batch_size, T_prime, 1)
        cde_input = torch.cat([time_channel, patched_features], dim=-1)
        coeffs = torchcde.natural_cubic_coeffs(cde_input, new_times)
        
    X = torchcde.CubicSpline(coeffs, new_times)
    z0_input = X.evaluate(new_times[0])
    
    # Differential learning rates: 1e-5 for CDE parameters, 1e-3 for Decoder parameters
    params = [
        {'params': encoder.initial_mapping.parameters(), 'lr': args.lr * 0.01},
        {'params': encoder.cde_func.parameters(), 'lr': args.lr * 0.01},
        {'params': decoder.parameters(), 'lr': args.lr}
    ]
    optimizer = torch.optim.AdamW(params)
    loss_fn = torch.nn.MSELoss()
    
    # Diagnostic: Verify device placement
    print("\n--- Diagnostic: Device Placement ---")
    print(f"Device set to: {device}")
    print(f"Input c_wf device: {c_wf.device}")
    print(f"Input c_t device: {c_t.device}")
    print(f"Spline X coefficients device: {coeffs.device}")
    print(f"Encoder initial_mapping parameter device: {next(encoder.initial_mapping.parameters()).device}")
    print(f"Encoder cde_func parameter device: {next(encoder.cde_func.parameters()).device}")
    print(f"Decoder parameter device: {next(decoder.parameters()).device}")
    print("------------------------------------\n")
    
    print(f"\n--- Overfitting Sample {args.sample_idx} (Joint Training, Differential LRs) for {args.epochs} Steps ---")
    decoder.train()
    
    import time
    for step in range(1, args.epochs + 1):
        step_start = time.time()
        optimizer.zero_grad()
        
        # CDE Integration gradients are now tracked
        z0 = encoder.initial_mapping(z0_input)
        z_t = torchcde.cdeint(X=X, z0=z0, func=encoder.cde_func, t=new_times, adjoint=False, method="rk4")
        
        # Interpolate from T_prime (62) to T (1000)
        T_target = c_t.shape[0]
        z_t_transposed = z_t.transpose(1, 2)
        z_t_interpolated = torch.nn.functional.interpolate(
            z_t_transposed, size=T_target, mode='linear', align_corners=True
        )
        z_t_final = z_t_interpolated.transpose(1, 2)
        
        recon_wf = decoder(z_t_final, c_t)
        loss = loss_fn(recon_wf, c_wf)
        loss.backward()
        optimizer.step()
        
        step_time = time.time() - step_start
        if step % 10 == 0 or step == 1:
            print(f"Step {step}/{args.epochs} | Loss: {loss.item():.5f} | Time: {step_time:.4f}s")
            
        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                recon_np_step = recon_wf[0, :, 1].cpu().numpy()
                orig_np_step = c_wf[0, :, 1].cpu().numpy()
                corr_step = float(compute_pearson_correlation(recon_wf, c_wf))
                
                images_dir = os.path.join("output", args.wandb_run_name, "images")
                os.makedirs(images_dir, exist_ok=True)
                step_plot_path = os.path.join(images_dir, f"overfit_{step}.png")
                save_overfit_plot(step, c_t.cpu().numpy(), orig_np_step, recon_np_step, step_plot_path, corr_step)
            
    # Evaluation
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        z_t_final = get_interpolated_latent_trajectory(encoder, c_wf, c_t)
        recon_final = decoder(z_t_final, c_t)
        
    print("\n--- Diagnostic: Latent Trajectory Statistics ---")
    print(f"Latent trajectory shape: {z_t_final.shape}")
    print(f"Latent trajectory overall mean: {z_t_final.mean().item():.4f}")
    print(f"Latent trajectory overall std: {z_t_final.std().item():.4f}")
    print(f"Latent trajectory temporal variation (mean std along time axis): {z_t_final.std(dim=1).mean().item():.4f}")
    print("------------------------------------------------\n")
    
    print("--- Running Coordinate-Only SIREN Overfitting Test ---")
    coord_decoder = PhaseTolerantDecoder(latent_dim=1, leads=leads, hidden_dim=args.decoder_hidden_dim).to(device)
    coord_optimizer = torch.optim.AdamW(coord_decoder.parameters(), lr=args.lr)
    
    # Scale times to [-1, 1] range for SIREN coordinate input
    t_min, t_max = c_t.min(), c_t.max()
    t_scaled = 2.0 * (c_t - t_min) / (t_max - t_min + 1e-8) - 1.0
    coord_input = t_scaled.unsqueeze(0).unsqueeze(-1) # [1, 1000, 1]
    
    coord_decoder.train()
    for step in range(1, args.epochs + 1):
        coord_optimizer.zero_grad()
        recon_coord = coord_decoder(coord_input, c_t)
        loss_coord = loss_fn(recon_coord, c_wf)
        loss_coord.backward()
        coord_optimizer.step()
        
        if step % 50 == 0 or step == 1:
            print(f"Step {step}/{args.epochs} | Coord Loss: {loss_coord.item():.5f}")
            
    coord_decoder.eval()
    with torch.no_grad():
        recon_coord_final = coord_decoder(coord_input, c_t)
        
    corr_coord = float(compute_pearson_correlation(recon_coord_final, c_wf))
    mse_coord = float(torch.nn.MSELoss()(recon_coord_final, c_wf).item())
    
    # Save coordinate results
    coord_plot_file = os.path.join("output", args.wandb_run_name, "overfit_coord_reconstruction.png")
    plt.figure(figsize=(12, 5))
    plt.plot(c_t.cpu().numpy(), context_wf[:, 1].cpu().numpy(), label="Original ECG (Lead II)", color="black", alpha=0.7)
    plt.plot(c_t.cpu().numpy(), recon_coord_final[0, :, 1].cpu().numpy(), label="Coord SIREN Reconstruction", color="blue", linestyle="--", alpha=0.9)
    plt.title(f"Coordinate-Only SIREN Fit (Pearson r: {corr_coord:.4f})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Normalized Voltage")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.savefig(coord_plot_file, dpi=150)
    plt.close()
    
    print("\n--- Coordinate-Only SIREN Overfitting Results ---")
    print(f"Final Coord MSE: {mse_coord:.5f}")
    print(f"Pearson r:       {corr_coord:.4f}")
    print(f"Plot saved to:   {coord_plot_file}")
    print("-------------------------------------------------\n")
        
    orig_np = context_wf[:, 1].cpu().numpy() # Lead II
    recon_np = recon_final[0, :, 1].cpu().numpy() # Lead II
    
    corr = float(compute_pearson_correlation(recon_final, c_wf))
    mse = float(torch.nn.MSELoss()(recon_final, c_wf).item())
    
    metrics = {
        "sample_index": args.sample_idx,
        "overfitting_steps": args.epochs,
        "final_loss": float(loss.item()),
        "pearson_correlation": corr,
        "mse": mse
    }
    rounded_metrics = round_dict_floats(metrics)
    
    # Save results
    output_dir = os.path.join("output", args.wandb_run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_file = os.path.join(output_dir, "overfit_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(rounded_metrics, f, indent=4)
        
    # Plotting comparison
    plt.figure(figsize=(12, 5))
    plt.plot(c_t.cpu().numpy(), orig_np, label="Original ECG (Lead II)", color="black", alpha=0.7)
    plt.plot(c_t.cpu().numpy(), recon_np, label="Overfit Reconstruction", color="red", linestyle="--", alpha=0.9)
    plt.title(f"Overfitting Sanity Check - Sample {args.sample_idx} (Pearson r: {rounded_metrics['pearson_correlation']:.4f})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Normalized Voltage")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    
    plot_file = os.path.join(output_dir, "overfit_reconstruction.png")
    plt.savefig(plot_file, dpi=150)
    plt.close()
    
    print("\n--- Overfitting Results ---")
    print(f"Final MSE:  {rounded_metrics['mse']:.5f}")
    print(f"Pearson r:  {rounded_metrics['pearson_correlation']:.4f}")
    print(f"Plot saved to:    {plot_file}")
    print(f"Metrics saved to: {metrics_file}")

if __name__ == "__main__":
    main()
