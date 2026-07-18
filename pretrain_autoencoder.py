import os
import argparse
import yaml
import random
import json
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from sde.encoder import PhysiologicalEncoder, get_interpolated_latent_trajectory
from sde.baseline import PhaseTolerantDecoder
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
    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    cov = (pred_centered * target_centered).sum(dim=1)
    pred_std = torch.sqrt((pred_centered ** 2).sum(dim=1))
    target_std = torch.sqrt((target_centered ** 2).sum(dim=1))
    eps = 1e-8
    r = cov / (pred_std * target_std + eps)
    r = torch.clamp(r, -1.0, 1.0)
    return r.mean().item()

def collate_fn(batch):
    context_wf = torch.stack([item[0] for item in batch])
    context_t = torch.stack([item[1] for item in batch])
    target_wf = torch.stack([item[2] for item in batch])
    target_t = torch.stack([item[3] for item in batch])
    segment_r_peaks = [item[4] for item in batch]
    return context_wf, context_t, target_wf, target_t, segment_r_peaks

def main():
    # 1. Configuration loading
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--config", type=str, default="config/pretrain_config.yaml", help="Path to config YAML file")
    temp_args, _ = temp_parser.parse_known_args()
    
    config_defaults = {}
    if os.path.exists(temp_args.config):
        try:
            with open(temp_args.config, "r") as f:
                config_defaults = yaml.safe_load(f)
            print(f"Loaded config from {temp_args.config}")
        except Exception as e:
            print(f"Failed to load config: {e}")
            
    parser = argparse.ArgumentParser(description="Pretrain CDE Encoder & SIREN Decoder")
    parser.add_argument("--config", type=str, default="config/pretrain_config.yaml")
    parser.add_argument("--db-dir", type=str, default=config_defaults.get("db_dir"))
    parser.add_argument("--weight-path", type=str, default=config_defaults.get("weight_path"))
    parser.add_argument("--pretrain-epochs", type=int, default=config_defaults.get("pretrain_epochs", 30), help="Number of pretraining epochs")
    parser.add_argument("--batch-size", type=int, default=config_defaults.get("batch_size", 128))
    parser.add_argument("--lr", type=float, default=config_defaults.get("lr", 1e-3), help="Decoder learning rate")
    parser.add_argument("--seed", type=int, default=config_defaults.get("seed", 42))
    parser.add_argument("--decoder-hidden-dim", type=int, default=config_defaults.get("decoder_hidden_dim", 256))
    parser.add_argument("--latent-dim", type=int, default=config_defaults.get("latent_dim", 32))
    parser.add_argument("--save-path", type=str, default=config_defaults.get("save_path", "checkpoints/pretrained_autoencoder.pt"))
    parser.add_argument("--resume", action="store_true", default=config_defaults.get("resume", False), help="Resume from previous checkpoint")
    parser.add_argument("--wandb-run-name", type=str, default=config_defaults.get("wandb_run_name", "autoencoder_pretrain"), help="Wandb run name")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Initialize Weights & Biases
    wandb.init(
        project="neuro-sde-baseline",
        name=args.wandb_run_name,
        config=vars(args),
        resume="allow" if args.resume else "never"
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 2. Datasets & Dataloaders
    train_recs, test_recs = get_incart_splits(args.db_dir, seed=args.seed)
    train_dataset = IncartDataset(args.db_dir, train_recs, use_cache=True)
    test_dataset = IncartDataset(args.db_dir, test_recs, use_cache=True)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    
    # Select a fixed random sample from the test set for tracking visual progress across epochs
    plot_idx = random.randint(0, len(test_dataset) - 1)
    print(f"Tracking visual reconstruction progress on random test sample index: {plot_idx}")
    
    # 3. Model instantiation
    encoder = PhysiologicalEncoder(in_leads=12, conv_layers=[(256, 2, 2)] * 4, latent_dim=args.latent_dim).to(device)
    decoder = PhaseTolerantDecoder(latent_dim=args.latent_dim, leads=12, hidden_dim=args.decoder_hidden_dim).to(device)
    
    # Load weights
    weight_file = args.weight_path or os.getenv("ECGFM_FT_WEIGHT_PATH")
    if weight_file:
        weight_file = os.path.expanduser(weight_file)
        if os.path.exists(weight_file):
            print(f"Loading pretrained weights into patcher: {weight_file}")
            load_pretrained_ecg_fm(encoder, weight_file)
            print("Freezing patcher CNN parameters.")
            for param in encoder.patcher.parameters():
                param.requires_grad = False
                
    # 4. Optimizer: 10x slower for CDE parameters, full rate for SIREN decoder
    encoder_params = [p for p in encoder.parameters() if p.requires_grad]
    params = [
        {'params': encoder_params, 'lr': args.lr * 0.1},
        {'params': decoder.parameters(), 'lr': args.lr}
    ]
    optimizer = torch.optim.AdamW(params)
    loss_fn = torch.nn.MSELoss()
    
    print("\n--- Starting Autoencoder Pretraining ---")
    best_loss = float("inf")
    start_epoch = 0
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    if args.resume and os.path.exists(args.save_path):
        print(f"Resuming from checkpoint: {args.save_path}...")
        checkpoint = torch.load(args.save_path, map_location=device)
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("val_loss", float("inf"))
        print(f"Resuming from epoch {start_epoch + 1} with best val loss: {best_loss:.5f}")
        
    for epoch in range(start_epoch, args.pretrain_epochs):
        encoder.train()
        decoder.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.pretrain_epochs}", leave=True)
        for batch in pbar:
            context_wf, context_t, target_wf, target_t, _ = batch
            context_wf = context_wf.to(device)
            context_t = context_t[0].to(device)
            
            optimizer.zero_grad()
            
            # Reconstruction only
            z_traj = get_interpolated_latent_trajectory(encoder, context_wf, context_t)
            recon_wf = decoder(z_traj, context_t)
            
            loss = loss_fn(recon_wf, context_wf)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} Train MSE Loss: {avg_train_loss:.5f}")
        
        # Validation evaluation
        encoder.eval()
        decoder.eval()
        val_loss = 0.0
        val_corr = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                context_wf, context_t, _, _, _ = batch
                context_wf = context_wf.to(device)
                context_t = context_t[0].to(device)
                
                z_traj = get_interpolated_latent_trajectory(encoder, context_wf, context_t)
                recon_wf = decoder(z_traj, context_t)
                
                val_loss += float(loss_fn(recon_wf, context_wf).item())
                val_corr += float(compute_pearson_correlation(recon_wf, context_wf))
                val_batches += 1
                
        avg_val_loss = val_loss / val_batches
        avg_val_corr = val_corr / val_batches
        print(f"Epoch {epoch+1} Val MSE Loss: {avg_val_loss:.5f} | Pearson r: {avg_val_corr:.4f}")
        
        # Log metrics to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train/mse_loss": avg_train_loss,
            "val/mse_loss": avg_val_loss,
            "val/pearson_correlation": avg_val_corr
        }, step=epoch+1)
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            state = {
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val_loss
            }
            torch.save(state, args.save_path)
            print(f"Saved best autoencoder model to {args.save_path}")
            
        # Generate reconstruction plot for the fixed random test sample
        with torch.no_grad():
            sample_wf, sample_t, _, _, _ = test_dataset[plot_idx]
            sample_wf_batch = sample_wf.unsqueeze(0).to(device)
            sample_t_device = sample_t.to(device)
            
            z_traj_sample = get_interpolated_latent_trajectory(encoder, sample_wf_batch, sample_t_device)
            recon_wf_sample = decoder(z_traj_sample, sample_t_device)
            
            sample_orig = sample_wf[:, 1].cpu().numpy() # Lead II
            sample_recon = recon_wf_sample[0, :, 1].cpu().numpy() # Lead II
            sample_t_np = sample_t.cpu().numpy()
            
            corr_sample = float(compute_pearson_correlation(recon_wf_sample, sample_wf_batch))
            
            plt.figure(figsize=(12, 5))
            plt.plot(sample_t_np, sample_orig, label="Original ECG (Lead II)", color="black", alpha=0.7)
            plt.plot(sample_t_np, sample_recon, label="Pretrained Recon", color="green", linestyle="--", alpha=0.9)
            plt.title(f"Pretraining Epoch {epoch+1} - Test Sample {plot_idx} (Pearson r: {corr_sample:.4f})")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Voltage")
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.legend()
            
            plot_dir = "output/pretraining"
            os.makedirs(plot_dir, exist_ok=True)
            plot_file = os.path.join(plot_dir, f"epoch_{epoch+1}.png")
            plt.savefig(plot_file, dpi=150)
            plt.close()
            
            # Log visual reconstruction plot to W&B
            wandb.log({"val/reconstruction_plot": wandb.Image(plot_file)}, step=epoch+1)
                
    print("\nPretraining Complete!")
    wandb.finish()

if __name__ == "__main__":
    main()
