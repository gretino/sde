import os
import argparse
import yaml
import random
import numpy as np
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
    temp_parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file")
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
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file")
    parser.add_argument("--db-dir", type=str, default="/home/qfbqt/8TB/datasets/physionet.org/files/incartdb/1.0.0", help="Path to database directory")
    parser.add_argument("--weight-path", type=str, default=None, help="Path to mimic_iv_ecg_finetuned.pt weight file")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick single step check")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--wandb-run-name", type=str, default="incart-epoch5", help="Weights & Biases run name")
    
    # Set default values from config file if present
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    # We evaluate test set with batch_size=1 to handle individual R-peak evaluation easily
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # 3. Instantiate model
    latent_dim = 32
    leads = 12
    conv_layers = [(256, 2, 2)] * 4
    
    encoder = PhysiologicalEncoder(in_leads=leads, conv_layers=conv_layers, latent_dim=latent_dim)
    f_base = FBase(latent_dim=latent_dim, hidden_dim=64)
    solver = ContinuousSolver(f_base=f_base)
    decoder = PhaseTolerantDecoder(latent_dim=latent_dim, leads=leads)
    
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
        
    # 5. Losses & Optimizer
    latent_loss_fn = LatentDynamicsLoss(model.encoder)
    waveform_loss_fn = PhaseTolerantWaveformLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
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
            recon_context_wf = model.decoder(z_context_traj)
            loss_recon_context = waveform_loss_fn(recon_context_wf, context_wf)

            z_target_traj = get_interpolated_latent_trajectory(model.encoder, target_wf, target_t)
            recon_target_wf = model.decoder(z_target_traj)
            loss_recon_target = waveform_loss_fn(recon_target_wf, target_wf)
            
            loss_ae = 0.5 * (loss_recon_context + loss_recon_target)
            
            # Weighted loss balancing: scale down waveform and autoencoder terms to match latent scale
            w_waveform = 0.05
            w_ae = 0.05
            loss = loss_latent + w_waveform * loss_waveform + w_ae * loss_ae
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
            
            predicted_latent = latent_trajectory[:, -1, :]
            loss_latent = latent_loss_fn(predicted_latent, target_wf, target_t)
            loss_waveform = waveform_loss_fn(predicted_wf, target_wf)
            
            # Metrics
            mse = nn.MSELoss()(predicted_wf, target_wf).item()
            mae = nn.L1Loss()(predicted_wf, target_wf).item()
            corr = compute_pearson_correlation(predicted_wf, target_wf)
            
            eval_latent_loss += loss_latent.item()
            eval_waveform_loss += loss_waveform.item()
            eval_mse += mse
            eval_mae += mae
            eval_correlation += corr
            
            # R-peak detection & physiology evaluation on Lead II (idx 1)
            # 1. Extract clean numpy signal for Lead II
            pred_wf_np = predicted_wf[0, :, 1].cpu().numpy()
            target_peaks = segment_r_peaks[0].cpu().numpy() # ground truth
            
            # 2. Extract peaks from predicted waveform via neurokit2
            try:
                # In case peak detection throws an error on flat/poor signals
                _, info = nk.ecg_peaks(pred_wf_np, sampling_rate=100)
                pred_peaks = info["ECG_R_Peaks"]
            except Exception:
                pred_peaks = np.array([])
                
            # Compute R-peak detection F1 (50ms tolerance = 5 samples)
            _, _, f1 = evaluate_peaks(pred_peaks, target_peaks, tol=5)
            f1_list.append(f1)
            
            # Compute HR / HRV RMSSD
            target_hr, target_rmssd = compute_hr_and_rmssd(target_peaks, sr=100)
            pred_hr, pred_rmssd = compute_hr_and_rmssd(pred_peaks, sr=100)
            
            hr_error_list.append(abs(target_hr - pred_hr))
            rmssd_error_list.append(abs(target_rmssd - pred_rmssd))
            
            pbar_eval.set_postfix({
                "mse": f"{mse:.4f}",
                "corr": f"{corr:.4f}",
                "f1": f"{f1:.4f}"
            })
            
            num_eval_samples += 1
            if args.dry_run and num_eval_samples >= 1:
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
