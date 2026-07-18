import os
import argparse
import yaml
import random
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import neurokit2 as nk
import wandb
from dotenv import load_dotenv
from tqdm import tqdm

from sde.encoder import PhysiologicalEncoder, get_interpolated_latent_trajectory
from sde.solver import ContinuousSolver, FBase
from sde.baseline import NeuroSDEBaseline, PhaseTolerantDecoder
from sde.loss import LatentDynamicsLoss, PhaseTolerantWaveformLoss
from sde.incart_dataset import IncartDataset, get_incart_splits
from sde.weight_utils import load_pretrained_ecg_fm

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_pearson_correlation(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes Pearson Correlation Coefficient averaged across batch and leads.
    """
    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    
    cov = (pred_centered * target_centered).sum(dim=1)
    
    pred_std = torch.sqrt((pred_centered ** 2).sum(dim=1))
    target_std = torch.sqrt((target_centered ** 2).sum(dim=1))
    
    eps = 1e-8
    r = cov / (pred_std * target_std + eps)
    # Clip to standard [-1, 1] range to avoid floating-point inaccuracies
    r = torch.clamp(r, -1.0, 1.0)
    return r.mean().item()

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

def evaluate_peaks(pred_peaks: np.ndarray, target_peaks: np.ndarray, tol: int = 5):
    """
    Computes precision, recall, and F1-score for R-peak alignment.
    Tolerance is in samples (e.g. 5 samples = 50ms at 100Hz).
    """
    if len(pred_peaks) == 0:
        return 0.0, 0.0, 0.0
        
    tp = 0
    matched_targets = set()
    for p in pred_peaks:
        min_dist = float("inf")
        closest_t = None
        for t in target_peaks:
            if t in matched_targets:
                continue
            dist = abs(p - t)
            if dist < min_dist:
                min_dist = dist
                closest_t = t
        if min_dist <= tol:
            tp += 1
            matched_targets.add(closest_t)
            
    precision = tp / len(pred_peaks) if len(pred_peaks) > 0 else 0.0
    recall = tp / len(target_peaks) if len(target_peaks) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0
    
    return precision, recall, f1

def compute_hr_and_rmssd(peaks: np.ndarray, sr: int = 100) -> tuple[float, float]:
    """
    Computes mean HR (BPM) and HRV RMSSD (ms) from peak sample indices.
    """
    if len(peaks) < 2:
        return 60.0, 0.0 # defaults
        
    rr_intervals = np.diff(peaks) / sr # RR in seconds
    mean_rr = np.mean(rr_intervals)
    hr = 60.0 / mean_rr
    
    if len(peaks) < 3:
        return hr, 0.0
        
    rr_diffs = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(rr_diffs ** 2)) * 1000.0 # in ms
    return hr, rmssd

def round_dict_floats(obj):
    if isinstance(obj, float):
        return round(obj, 4)
    elif isinstance(obj, dict):
        return {k: round_dict_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_dict_floats(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return round_dict_floats(obj.tolist())
    elif isinstance(obj, (np.float32, np.float64)):
        return round(float(obj), 4)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj

def collate_fn(batch):
    context_wf = torch.stack([item[0] for item in batch])
    context_t = torch.stack([item[1] for item in batch])
    target_wf = torch.stack([item[2] for item in batch])
    target_t = torch.stack([item[3] for item in batch])
    segment_r_peaks = [item[4] for item in batch]
    return context_wf, context_t, target_wf, target_t, segment_r_peaks

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
            print(f"Loaded training parameters from config: {temp_args.config}")
        except Exception as e:
            print(f"Warning: Failed to load config from {temp_args.config}: {e}")
            
    # 2. Main parser
    parser = argparse.ArgumentParser(description="Train and evaluate Neuro SDE on INCART 12-lead ECG dataset.")
    parser.add_argument("--config", type=str, default="config/incart_config.yaml", help="Path to config YAML file")
    parser.add_argument("--db-dir", type=str, default="/home/qfbqt/8TB/datasets/physionet.org/files/incartdb/1.0.0", help="Path to database directory")
    parser.add_argument("--weight-path", type=str, default=None, help="Path to mimic_iv_ecg_finetuned.pt weight file")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick single step check")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--wandb-run-name", type=str, default="incart-epoch5", help="Weights & Biases run name")
    parser.add_argument("--decoder-hidden-dim", type=int, default=256, help="Intermediate hidden dimension of the decoder")
    parser.add_argument("--latent-dim", type=int, default=32, help="Dimensionality of the continuous latent space")
    parser.add_argument("--pretrained-ae-path", type=str, default=None, help="Path to pretrained autoencoder checkpoint")
    parser.add_argument("--w-waveform", type=float, default=1.0, help="Weight for future waveform reconstruction loss")
    parser.add_argument("--w-ae", type=float, default=1.0, help="Weight for autoencoder reconstruction loss")
    parser.add_argument("--ae-lr", type=float, default=3e-5, help="Learning rate for decoder when fine-tuning/pretraining")
    parser.add_argument("--ae-encoder-lr", type=float, default=None, help="Learning rate for encoder when fine-tuning/pretraining (defaults to ae-lr * 0.1)")
    parser.add_argument("--resume-sde-path", type=str, default=None, help="Path to full NeuroSDE checkpoint to resume training from")

    
    # Set default values from config file if present
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    
    if args.ae_encoder_lr is None:
        args.ae_encoder_lr = args.ae_lr * 0.1
        
    set_seed(args.seed)

    
    # Initialize wandb using the config run name (defaults to "incart-epoch5")
    wandb.init(project="sde", config=vars(args), name=args.wandb_run_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load splits
    train_recs, test_recs = get_incart_splits(args.db_dir, seed=args.seed)
    print(f"Dataset splits: {len(train_recs)} train records, {len(test_recs)} test records")
    
    # 2. Build Datasets
    print("Loading and preprocessing datasets...")
    train_dataset = IncartDataset(args.db_dir, train_recs, use_cache=True)
    test_dataset = IncartDataset(args.db_dir, test_recs, use_cache=True)
    print(f"Dataset sizes: {len(train_dataset)} train segments, {len(test_dataset)} test segments")
    plot_idx = random.randint(0, len(test_dataset) - 1)
    print(f"Tracking visual prediction progress on test sample index: {plot_idx}")

    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    # We evaluate test set in batches of 128 for GPU efficiency
    test_loader = DataLoader(
        test_dataset, 
        batch_size=128, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # 3. Instantiate model
    latent_dim = args.latent_dim
    leads = 12
    conv_layers = [(256, 2, 2)] * 4
    
    encoder = PhysiologicalEncoder(in_leads=leads, conv_layers=conv_layers, latent_dim=latent_dim)
    f_base = FBase(latent_dim=latent_dim, hidden_dim=64)
    solver = ContinuousSolver(f_base=f_base)
    decoder = PhaseTolerantDecoder(latent_dim=latent_dim, leads=leads, hidden_dim=args.decoder_hidden_dim)
    
    model = NeuroSDEBaseline(encoder, solver, decoder).to(device)
    
    # 4. Load weights
    weight_file = args.weight_path or os.getenv("ECGFM_FT_WEIGHT_PATH")
    if weight_file:
        weight_file = os.path.expanduser(weight_file)
        if os.path.exists(weight_file):
            print(f"Loading pretrained weights from {weight_file}...")
            load_pretrained_ecg_fm(model.encoder, weight_file)
        else:
            print(f"Warning: Pretrained weight file not found at {weight_file}")
    else:
        print("No pretrained weights specified. Training from scratch.")
        
    # Load pretrained autoencoder if specified
    pretrained_ae_loaded = False
    if args.pretrained_ae_path:
        args.pretrained_ae_path = os.path.expanduser(args.pretrained_ae_path)
        if os.path.exists(args.pretrained_ae_path):
            print(f"Loading pretrained autoencoder from {args.pretrained_ae_path}...")
            checkpoint = torch.load(args.pretrained_ae_path, map_location=device)
            model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
            pretrained_ae_loaded = True
        else:
            print(f"Warning: Pretrained autoencoder file not found at {args.pretrained_ae_path}")
            
    # Load full SDE model checkpoint if specified (to resume/stage-2 train)
    if args.resume_sde_path:
        args.resume_sde_path = os.path.expanduser(args.resume_sde_path)
        if os.path.exists(args.resume_sde_path):
            print(f"Resuming SDE from full checkpoint: {args.resume_sde_path}...")
            checkpoint = torch.load(args.resume_sde_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print(f"Warning: Full NeuroSDE checkpoint not found at {args.resume_sde_path}")

            
    # 5. Losses & Optimizer
    latent_loss_fn = LatentDynamicsLoss(model.encoder)
    waveform_loss_fn = PhaseTolerantWaveformLoss()
    
    if pretrained_ae_loaded:
        if args.ae_lr > 0:
            print(f"Fine-tuning pretrained encoder (lr: {args.ae_encoder_lr}) and decoder (lr: {args.ae_lr}) parameters.")
            for param in model.encoder.parameters():
                param.requires_grad = True
            for param in model.decoder.parameters():
                param.requires_grad = True
                
            # Keep patcher frozen if pretrained weights were loaded for it
            if weight_file:
                print("Freezing pretrained encoder patcher parameters.")
                for param in model.encoder.patcher.parameters():
                    param.requires_grad = False
                    
            encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
            decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
            params = [
                {'params': encoder_params, 'lr': args.ae_encoder_lr},
                {'params': model.solver.parameters(), 'lr': args.lr},
                {'params': decoder_params, 'lr': args.ae_lr}
            ]

        else:
            # Freeze the pretrained encoder and decoder to preserve reconstruction and isolate dynamics learning
            print("Freezing pretrained encoder and decoder parameters for SDE solver pretraining.")
            for param in model.encoder.parameters():
                param.requires_grad = False
            for param in model.decoder.parameters():
                param.requires_grad = False
                
            params = [
                {'params': model.solver.parameters(), 'lr': args.lr}
            ]

    else:
        # Freeze the CNN patcher parameters if we loaded pretrained weights
        if weight_file:
            print("Freezing pretrained encoder patcher parameters.")
            for param in model.encoder.patcher.parameters():
                param.requires_grad = False
                
        # Differential learning rates: 10x smaller for continuous CDE/SDE dynamics, full rate for SIREN decoder
        encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
        params = [
            {'params': encoder_params, 'lr': args.lr * 0.1},
            {'params': model.solver.parameters(), 'lr': args.lr * 0.1},
            {'params': model.decoder.parameters(), 'lr': args.lr}
        ]
        
    optimizer = torch.optim.AdamW(params)
    
    # 6. Training Loop
    print("\nStarting Training...")
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_latent_loss = 0.0
        epoch_waveform_loss = 0.0
        epoch_ae_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        for batch in pbar:
            context_wf, context_t, target_wf, target_t, _ = batch
            
            context_wf = context_wf.to(device)
            context_t = context_t[0].to(device) # common timestamps
            target_wf = target_wf.to(device)
            target_t = target_t[0].to(device)
            
            optimizer.zero_grad()
            
            predicted_wf, latent_trajectory = model(context_wf, context_t, target_t)
            
            # Compute losses
            # Latent Dynamics compares extrapolated state with target encoded representation
            predicted_latent = latent_trajectory[:, -1, :] # latent at target end time
            loss_latent = latent_loss_fn(predicted_latent, target_wf, target_t)
            
            # Reconstruction compares waveform output
            loss_waveform = waveform_loss_fn(predicted_wf, target_wf)
            
            # Direct autoencoder reconstruction losses to ground the decoder
            z_context_traj = get_interpolated_latent_trajectory(model.encoder, context_wf, context_t)
            recon_context_wf = model.decoder(z_context_traj, context_t)
            loss_recon_context = waveform_loss_fn(recon_context_wf, context_wf)

            z_target_traj = get_interpolated_latent_trajectory(model.encoder, target_wf, target_t)
            recon_target_wf = model.decoder(z_target_traj, target_t)
            loss_recon_target = waveform_loss_fn(recon_target_wf, target_wf)
            
            loss_ae = 0.5 * (loss_recon_context + loss_recon_target)
            
            # Both latent and waveform losses now naturally share the same scale (~18)
            loss = loss_latent + args.w_waveform * loss_waveform + args.w_ae * loss_ae
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_latent_loss += loss_latent.item()
            epoch_waveform_loss += loss_waveform.item()
            epoch_ae_loss += loss_ae.item()
            num_batches += 1
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "latent": f"{loss_latent.item():.4f}",
                "waveform": f"{loss_waveform.item():.4f}",
                "ae": f"{loss_ae.item():.4f}"
            })
            
            if args.dry_run:
                break
                
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss/num_batches:.4f} "
              f"(Latent: {epoch_latent_loss/num_batches:.4f}, Waveform: {epoch_waveform_loss/num_batches:.4f}, AE: {epoch_ae_loss/num_batches:.4f})")
              
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": epoch_loss / num_batches,
            "train/latent_loss": epoch_latent_loss / num_batches,
            "train/waveform_loss": epoch_waveform_loss / num_batches,
            "train/ae_loss": epoch_ae_loss / num_batches
        })
        
        # Generate prediction plot every 2 epochs (or on dry-run for verification)
        if (epoch + 1) % 2 == 0 or args.dry_run:
            model.eval()
            with torch.no_grad():
                sample = test_dataset[plot_idx]
                context_wf, context_t, target_wf, target_t, _ = sample
                
                c_wf = context_wf.unsqueeze(0).to(device)
                c_t = context_t.to(device)
                t_wf = target_wf.unsqueeze(0).to(device)
                t_t = target_t.to(device)
                
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
                
                # Generate ECG Comparison Plot (3-panel)
                fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharey=True)
                
                # Panel 1: Context Window (Past 10 seconds)
                axes[0].plot(context_t.cpu().numpy(), orig_context_np, label="Original Context ECG", color="black", alpha=0.7)
                axes[0].plot(context_t.cpu().numpy(), recon_context_np, label="Decoder Reconstruction", color="blue", linestyle="--", alpha=0.9)
                axes[0].set_title(f"Context Window (Past 10s) - Lead II - (Pearson r: {corr_context:.4f})")
                axes[0].set_ylabel("Normalized Voltage")
                axes[0].grid(True, linestyle=":", alpha=0.6)
                axes[0].legend(loc="upper right")
                
                # Panel 2: Future Window Panel - Raw (Future 10 seconds)
                axes[1].plot(target_t.cpu().numpy(), orig_target_np, label="Original Target ECG", color="black", alpha=0.7)
                axes[1].plot(target_t.cpu().numpy(), recon_oracle_np, label="Oracle Decoder Reconstruction", color="green", linestyle=":", alpha=0.9)
                axes[1].plot(target_t.cpu().numpy(), recon_sde_np, label="Raw SDE Evolved Forecast", color="red", linestyle="--", alpha=0.9)
                axes[1].set_title(f"Future Window (Next 10s) - Lead II - Raw Forecast (Oracle r: {corr_oracle:.4f} | SDE Forecast r: {corr_sde:.4f})")
                axes[1].set_ylabel("Normalized Voltage")
                axes[1].grid(True, linestyle=":", alpha=0.6)
                axes[1].legend(loc="upper right")
                
                # Panel 3: Future Window Panel - Phase-Aligned (Future 10 seconds)
                axes[2].plot(target_t.cpu().numpy(), orig_target_np, label="Original Target ECG", color="black", alpha=0.7)
                axes[2].plot(target_t.cpu().numpy(), recon_sde_shifted_np, label=f"Phase-Aligned SDE Forecast (Shift: {shift_seconds:.2f}s)", color="purple", linestyle="--", alpha=0.9)
                axes[2].set_title(f"Future Window (Next 10s) - Lead II - Phase-Aligned Forecast (Aligned r: {corr_sde_shifted:.4f})")
                axes[2].set_xlabel("Time (seconds)")
                axes[2].set_ylabel("Normalized Voltage")
                axes[2].grid(True, linestyle=":", alpha=0.6)
                axes[2].legend(loc="upper right")
                
                plt.tight_layout()
                plot_dir = os.path.join("output", args.wandb_run_name, "plots")
                os.makedirs(plot_dir, exist_ok=True)
                plot_file = os.path.join(plot_dir, f"epoch_{epoch+1}.png")
                plt.savefig(plot_file, dpi=150)
                plt.close()
                
                wandb.log({"val/prediction_plot": wandb.Image(plot_file)}, step=epoch+1)
                print(f"Comparison plot saved to {plot_file}")
                
            model.train()
            
        if args.dry_run:
            break

            
    # Save the model checkpoint
    if not args.dry_run:
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, "neurosde_final.pt")
        torch.save({
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, save_path)
        print(f"\nModel checkpoint saved to {save_path}")
            
    # 7. Evaluation
    print("\nStarting Evaluation...")
    model.eval()
    
    eval_latent_loss = 0.0
    eval_waveform_loss = 0.0
    eval_mse = 0.0
    eval_mae = 0.0
    eval_correlation = 0.0
    
    # Physiological lists
    f1_list = []
    hr_error_list = []
    rmssd_error_list = []
    
    num_eval_samples = 0
    
    pbar_eval = tqdm(test_loader, desc="Evaluating", leave=True)
    with torch.no_grad():
        for batch in pbar_eval:
            context_wf, context_t, target_wf, target_t, segment_r_peaks = batch
            
            context_wf = context_wf.to(device)
            context_t = context_t[0].to(device)
            target_wf = target_wf.to(device)
            target_t = target_t[0].to(device)
            
            predicted_wf, latent_trajectory = model(context_wf, context_t, target_t)
            
            batch_size = context_wf.shape[0]
            for b in range(batch_size):
                single_pred_wf = predicted_wf[b:b+1]
                single_target_wf = target_wf[b:b+1]
                
                single_latent = latent_trajectory[b:b+1, -1, :]
                loss_latent = latent_loss_fn(single_latent, single_target_wf, target_t)
                loss_waveform = waveform_loss_fn(single_pred_wf, single_target_wf)
                
                # Metrics
                mse = nn.MSELoss()(single_pred_wf, single_target_wf).item()
                mae = nn.L1Loss()(single_pred_wf, single_target_wf).item()
                corr = compute_pearson_correlation(single_pred_wf, single_target_wf)
                
                eval_latent_loss += loss_latent.item()
                eval_waveform_loss += loss_waveform.item()
                eval_mse += mse
                eval_mae += mae
                eval_correlation += corr
                
                # R-peak detection & physiology evaluation on Lead II (idx 1)
                pred_wf_np = single_pred_wf[0, :, 1].cpu().numpy()
                target_peaks = segment_r_peaks[b].cpu().numpy()
                
                try:
                    _, info = nk.ecg_peaks(pred_wf_np, sampling_rate=100)
                    pred_peaks = info["ECG_R_Peaks"]
                except Exception:
                    pred_peaks = np.array([])
                    
                _, _, f1 = evaluate_peaks(pred_peaks, target_peaks, tol=5)
                f1_list.append(f1)
                
                target_hr, target_rmssd = compute_hr_and_rmssd(target_peaks, sr=100)
                pred_hr, pred_rmssd = compute_hr_and_rmssd(pred_peaks, sr=100)
                
                hr_error_list.append(abs(target_hr - pred_hr))
                rmssd_error_list.append(abs(target_rmssd - pred_rmssd))
                
                num_eval_samples += 1
                
            pbar_eval.set_postfix({
                "mse": f"{eval_mse/num_eval_samples:.4f}",
                "corr": f"{eval_correlation/num_eval_samples:.4f}",
                "f1": f"{np.mean(f1_list):.4f}"
            })
            
            if args.dry_run:
                break
                
    # 8. Report metrics
    print("\n--- Numerical Research Metrics on INCART Test Set ---")
    print(f"Total evaluated segments: {num_eval_samples}")
    print(f"Latent Dynamics Loss (MSE): {eval_latent_loss/num_eval_samples:.5f}")
    print(f"Waveform Loss (Composite):  {eval_waveform_loss/num_eval_samples:.5f}")
    print(f"Reconstruction MSE:         {eval_mse/num_eval_samples:.5f}")
    print(f"Reconstruction MAE:         {eval_mae/num_eval_samples:.5f}")
    print(f"Pearson Correlation (r):    {eval_correlation/num_eval_samples:.4f}")
    print(f"R-peak Detection F1-Score:  {np.mean(f1_list):.4f}")
    print(f"Heart Rate MAE (BPM):       {np.mean(hr_error_list):.2f}")
    print(f"HRV RMSSD MAE (ms):         {np.mean(rmssd_error_list):.2f}")
    print("-----------------------------------------------------")
    
    # Save metrics to output/[run_name]/metrics.json
    metrics_dict = {
        "total_evaluated_segments": int(num_eval_samples),
        "latent_dynamics_loss_mse": float(eval_latent_loss / num_eval_samples),
        "waveform_loss_composite": float(eval_waveform_loss / num_eval_samples),
        "reconstruction_mse": float(eval_mse / num_eval_samples),
        "reconstruction_mae": float(eval_mae / num_eval_samples),
        "pearson_correlation": float(eval_correlation / num_eval_samples),
        "rpeak_detection_f1": float(np.mean(f1_list)),
        "heart_rate_mae_bpm": float(np.mean(hr_error_list)),
        "hrv_rmssd_mae_ms": float(np.mean(rmssd_error_list))
    }
    rounded_metrics = round_dict_floats(metrics_dict)
    
    run_name = args.wandb_run_name
    output_dir = os.path.join("output", run_name)
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, "metrics.json")
    
    with open(metrics_file, "w") as f:
        json.dump(rounded_metrics, f, indent=4)
    print(f"Metrics successfully written to {metrics_file}")
    
    wandb.log({
        "val/latent_loss": eval_latent_loss / num_eval_samples,
        "val/waveform_loss": eval_waveform_loss / num_eval_samples,
        "val/reconstruction_mse": eval_mse / num_eval_samples,
        "val/reconstruction_mae": eval_mae / num_eval_samples,
        "val/pearson_correlation": eval_correlation / num_eval_samples,
        "val/rpeak_f1": np.mean(f1_list),
        "val/hr_mae": np.mean(hr_error_list),
        "val/rmssd_mae": np.mean(rmssd_error_list)
    })
    
    wandb.finish()

if __name__ == "__main__":
    main()
