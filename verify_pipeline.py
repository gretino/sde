import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import neurokit2 as nk
import matplotlib.pyplot as plt
import json
import yaml
import torchcde
from dotenv import load_dotenv

from sde.encoder import PhysiologicalEncoder, get_interpolated_latent_trajectory
from sde.solver import ContinuousSolver, FBase
from sde.baseline import NeuroSDEBaseline, PhaseTolerantDecoder
from sde.loss import LatentDynamicsLoss, PhaseTolerantWaveformLoss
from sde.incart_dataset import IncartDataset, get_incart_splits
from sde.weight_utils import load_pretrained_ecg_fm
from run_incart_experiment import set_seed, compute_pearson_correlation, evaluate_peaks, compute_hr_and_rmssd, collate_fn

# Define MLP for Latent Dynamics Baselines
class MLP(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, z: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        # z: [batch, latent_dim]
        # dt: [batch]
        if len(dt.shape) == 1:
            dt = dt.unsqueeze(-1)
        x = torch.cat([z, dt], dim=-1)
        return self.net(x)


def evaluate_waveform_predictions(preds, targets, segment_r_peaks_list, sr=100):
    mse_list = []
    mae_list = []
    corr_list = []
    f1_list = []
    hr_err_list = []
    rmssd_err_list = []
    
    for i in range(len(preds)):
        pred = preds[i]
        target = targets[i]
        peaks = segment_r_peaks_list[i].cpu().numpy()
        
        pred_flat = pred.squeeze(0) # [1000, 12]
        target_flat = target.squeeze(0) # [1000, 12]
        
        mse = nn.MSELoss()(pred_flat, target_flat).item()
        mae = nn.L1Loss()(pred_flat, target_flat).item()
        corr = compute_pearson_correlation(pred, target)
        
        mse_list.append(mse)
        mae_list.append(mae)
        corr_list.append(corr)
        
        # Peaks evaluation on Lead II (idx 1)
        pred_wf_np = pred_flat[:, 1].cpu().numpy()
        try:
            _, info = nk.ecg_peaks(pred_wf_np, sampling_rate=sr)
            pred_peaks = info["ECG_R_Peaks"]
        except Exception:
            pred_peaks = np.array([])
            
        _, _, f1 = evaluate_peaks(pred_peaks, peaks, tol=5)
        f1_list.append(f1)
        
        target_hr, target_rmssd = compute_hr_and_rmssd(peaks, sr=sr)
        pred_hr, pred_rmssd = compute_hr_and_rmssd(pred_peaks, sr=sr)
        
        hr_err_list.append(abs(target_hr - pred_hr))
        rmssd_err_list.append(abs(target_rmssd - pred_rmssd))
        
    return {
        "mse": float(np.mean(mse_list)),
        "mae": float(np.mean(mae_list)),
        "pearson_correlation": float(np.mean(corr_list)),
        "rpeak_f1": float(np.mean(f1_list)),
        "hr_mae": float(np.mean(hr_err_list)),
        "rmssd_mae": float(np.mean(rmssd_err_list))
    }

def compute_latent_metrics(pred, target, z0=None):
    mse = nn.MSELoss()(pred, target).item()
    mae = nn.L1Loss()(pred, target).item()
    
    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean().item()
    
    # R^2 against persistence baseline
    r2 = 0.0
    if z0 is not None:
        mse_persist = nn.MSELoss()(z0, target).item()
        if mse_persist > 1e-8:
            r2 = 1.0 - (mse / mse_persist)
            
    return {"mse": float(mse), "mae": float(mae), "cosine_similarity": float(cos_sim), "r2_vs_persistence": float(r2)}

def plot_and_save_ecg(original, reconstructed, original_peaks, reconstructed_peaks, save_path, title):
    plt.figure(figsize=(12, 4))
    t = np.arange(len(original)) / 100.0 # 100Hz
    
    plt.plot(t, original, label="Original / Target", color="#1f77b4", alpha=0.8)
    plt.plot(t, reconstructed, label="Reconstructed / Predicted", color="#ff7f0e", linestyle="--", alpha=0.8)
    
    if len(original_peaks) > 0:
        plt.scatter(original_peaks / 100.0, original[original_peaks], color="blue", marker="o", s=40, label="Original R-peaks")
    if len(reconstructed_peaks) > 0:
        plt.scatter(reconstructed_peaks / 100.0, reconstructed[reconstructed_peaks], color="red", marker="x", s=50, label="Reconstructed R-peaks")
        
    plt.title(title, fontsize=12)
    plt.xlabel("Time (s)", fontsize=10)
    plt.ylabel("Normalized Voltage", fontsize=10)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

@torch.no_grad()
def extract_latent_pairs(encoder, dataloader, device, max_samples=None):
    encoder.eval()
    all_z0 = []
    all_z_dt = {t: [] for t in range(1, 11)}
    
    count = 0
    for batch in dataloader:
        context_wf, context_t, target_wf, target_t, _ = batch
        context_wf = context_wf.to(device)
        context_t = context_t[0].to(device)
        target_wf = target_wf.to(device)
        target_t = target_t[0].to(device)
        
        # 1. Extract z0
        z0 = encoder(context_wf, context_t)
        all_z0.append(z0.cpu())
        
        # 2. Extract z_dt for each dt
        for dt_val in range(1, 11):
            idx = dt_val * 100 # at 100Hz
            sliced_wf = target_wf[:, :idx, :]
            sliced_t = target_t[:idx]
            z_dt = encoder(sliced_wf, sliced_t)
            all_z_dt[dt_val].append(z_dt.cpu())
            
        count += context_wf.size(0)
        if max_samples and count >= max_samples:
            break
            
    z0_tensor = torch.cat(all_z0, dim=0)
    z_dt_tensors = {dt_val: torch.cat(all_z_dt[dt_val], dim=0) for dt_val in range(1, 11)}
    return z0_tensor, z_dt_tensors

def train_latent_baselines(z0_train, z_dt_train_dict, device, epochs=10, batch_size=64):
    latent_dim = z0_train.size(1)
    
    # 1. Instantiate models
    linear_models = nn.ModuleDict({
        str(dt): nn.Linear(latent_dim, latent_dim) for dt in range(1, 11)
    }).to(device)
    
    mlp_model = MLP(latent_dim=latent_dim, hidden_dim=64).to(device)
    
    # 2. Setup optimizers
    linear_optimizer = torch.optim.AdamW(linear_models.parameters(), lr=1e-3)
    mlp_optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    # Prepare training data for MLP
    mlp_inputs_z0 = []
    mlp_inputs_dt = []
    mlp_targets = []
    for dt_val in range(1, 11):
        mlp_inputs_z0.append(z0_train)
        mlp_inputs_dt.append(torch.full((z0_train.size(0),), float(dt_val)))
        mlp_targets.append(z_dt_train_dict[dt_val])
        
    mlp_z0 = torch.cat(mlp_inputs_z0, dim=0)
    mlp_dt = torch.cat(mlp_inputs_dt, dim=0)
    mlp_tgt = torch.cat(mlp_targets, dim=0)
    
    num_samples = mlp_z0.size(0)
    
    # Train Linear and MLP models
    for epoch in range(epochs):
        # --- Train MLP ---
        mlp_model.train()
        indices = torch.randperm(num_samples)
        epoch_mlp_loss = 0.0
        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            b_z0 = mlp_z0[batch_idx].to(device)
            b_dt = mlp_dt[batch_idx].to(device)
            b_tgt = mlp_tgt[batch_idx].to(device)
            
            mlp_optimizer.zero_grad()
            pred = mlp_model(b_z0, b_dt)
            loss = loss_fn(pred, b_tgt)
            loss.backward()
            mlp_optimizer.step()
            epoch_mlp_loss += loss.item() * b_z0.size(0)
            
        # --- Train Linear models ---
        linear_models.train()
        epoch_linear_loss = 0.0
        linear_indices = torch.randperm(z0_train.size(0))
        for dt_val in range(1, 11):
            dt_z0 = z0_train.to(device)
            dt_tgt = z_dt_train_dict[dt_val].to(device)
            
            for i in range(0, z0_train.size(0), batch_size):
                batch_idx = linear_indices[i:i+batch_size]
                b_z0 = dt_z0[batch_idx]
                b_tgt = dt_tgt[batch_idx]
                
                linear_optimizer.zero_grad()
                pred = linear_models[str(dt_val)](b_z0)
                loss = loss_fn(pred, b_tgt)
                loss.backward()
                linear_optimizer.step()
                epoch_linear_loss += loss.item() * b_z0.size(0)
                
        print(f"Baseline Epoch {epoch+1}/{epochs} - MLP Loss: {epoch_mlp_loss/num_samples:.5f}, Linear Avg Loss: {epoch_linear_loss/(10*z0_train.size(0)):.5f}")
        
    return linear_models, mlp_model

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

def save_json_metrics(results, args):
    run_name = args.wandb_run_name
    output_dir = os.path.join("output", run_name)
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, "metrics.json")
    
    rounded_results = round_dict_floats(results)
    
    with open(metrics_file, "w") as f:
        json.dump(rounded_results, f, indent=4)
        
    print(f"Metrics successfully saved to {metrics_file}")

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
    parser = argparse.ArgumentParser(description="Verification and Baselines Suite for Neuro SDE Pipeline")
    parser.add_argument("--config", type=str, default="config/incart_config.yaml", help="Path to config YAML file")
    parser.add_argument("--db-dir", type=str, default="/home/qfbqt/8TB/datasets/physionet.org/files/incartdb/1.0.0", help="Path to database directory")
    parser.add_argument("--weight-path", type=str, default=None, help="Path to mimic_iv_ecg_finetuned.pt weight file")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/stage1/neurosde_final.pt", help="Path to SDE checkpoint")
    parser.add_argument("--save-dir", type=str, default="verification_results", help="Directory to save metrics and plots")
    parser.add_argument("--max-samples", type=int, default=100, help="Maximum test samples to evaluate for slow metrics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb-run-name", type=str, default="incart-epoch5", help="Run name used for output folder")
    parser.add_argument("--decoder-hidden-dim", type=int, default=256, help="Intermediate hidden dimension of the decoder")
    parser.add_argument("--latent-dim", type=int, default=32, help="Dimensionality of the continuous latent space")
    
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "plots"), exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Splits and Datasets
    train_recs, test_recs = get_incart_splits(args.db_dir, seed=args.seed)
    train_dataset = IncartDataset(args.db_dir, train_recs, use_cache=True)
    test_dataset = IncartDataset(args.db_dir, test_recs, use_cache=True)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=128, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=128, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # 2. Instantiate Model and Load Pretrained Encoder Weights
    latent_dim = args.latent_dim
    leads = 12
    conv_layers = [(256, 2, 2)] * 4
    
    encoder = PhysiologicalEncoder(in_leads=leads, conv_layers=conv_layers, latent_dim=latent_dim)
    f_base = FBase(latent_dim=latent_dim, hidden_dim=64)
    solver = ContinuousSolver(f_base=f_base)
    decoder = PhaseTolerantDecoder(latent_dim=latent_dim, leads=leads, hidden_dim=args.decoder_hidden_dim)
    model = NeuroSDEBaseline(encoder, solver, decoder).to(device)
    
    weight_file = args.weight_path or os.getenv("ECGFM_FT_WEIGHT_PATH")
    if weight_file:
        weight_file = os.path.expanduser(weight_file)
        if os.path.exists(weight_file):
            print(f"Loading pretrained encoder weights from {weight_file}...")
            load_pretrained_ecg_fm(model.encoder, weight_file)
        else:
            print(f"Warning: Pretrained weight file not found at {weight_file}")
            
    results = {}
    
    # ==========================================
    # SECTION 1: Verify Dataset and Preprocessing
    # ==========================================
    print("\n--- Running Section 1: Verify Dataset and Preprocessing ---")
    lead_ok = True
    sampling_ok = True
    normalization_ok = True
    alignment_ok = True
    
    for idx in range(min(5, len(test_dataset))):
        context_wf, context_t, target_wf, target_t, segment_r_peaks = test_dataset[idx]
        # Test lead correctness (INCART must have 12 leads)
        if context_wf.shape[1] != 12:
            lead_ok = False
        # Test shape
        if len(context_wf.shape) != 2:
            lead_ok = False
        # Test sampling rate (10.0s window should have 1000 points at 100Hz)
        if context_wf.shape[0] != 1000 or target_wf.shape[0] != 1000:
            sampling_ok = False
        # Test normalization (mean approx 0, std approx 1 per lead)
        for l in range(12):
            m = context_wf[:, l].mean().item()
            s = context_wf[:, l].std().item()
            if abs(m) > 0.05 or abs(s - 1.0) > 0.05:
                normalization_ok = False
        # Test target alignment (target starts after context)
        if target_t[0].item() - context_t[-1].item() <= 0:
            alignment_ok = False
            
    print(f"Lead Shape Correctness check: {'PASSED' if lead_ok else 'FAILED'}")
    print(f"Sampling Correctness check: {'PASSED' if sampling_ok else 'FAILED'}")
    print(f"Normalization Correctness check: {'PASSED' if normalization_ok else 'FAILED'}")
    print(f"Target Alignment check: {'PASSED' if alignment_ok else 'FAILED'}")
    
    results["section_1"] = {
        "lead_shape_passed": lead_ok,
        "sampling_passed": sampling_ok,
        "normalization_passed": normalization_ok,
        "alignment_passed": alignment_ok
    }
    
    # ==========================================
    # SECTION 2: Test ECG-FM Encoder Output
    # ==========================================
    print("\n--- Running Section 2: Test ECG-FM Encoder Output ---")
    model.eval()
    
    # Test stability
    context_wf, context_t, _, _, _ = test_dataset[0]
    c_wf = context_wf.unsqueeze(0).to(device)
    c_t = context_t.to(device)
    with torch.no_grad():
        z1 = model.encoder(c_wf, c_t)
        z2 = model.encoder(c_wf, c_t)
    stability_passed = torch.allclose(z1, z2)
    print(f"Encoder stability check (eval mode): {'PASSED' if stability_passed else 'FAILED'}")
    
    # Statistics extraction
    print("Extracting test set latent statistics...")
    test_latents = []
    eval_batch_size = min(32, args.max_samples)
    eval_loader = DataLoader(
        test_dataset, 
        batch_size=eval_batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    samples_processed = 0
    with torch.no_grad():
        for batch in eval_loader:
            if samples_processed >= args.max_samples:
                break
            
            c_wf = batch[0].to(device)
            c_t = batch[1][0].to(device)
            z = model.encoder(c_wf, c_t)
            
            remaining = args.max_samples - samples_processed
            actual_take = min(z.shape[0], remaining)
            test_latents.append(z[:actual_take].cpu())
            samples_processed += actual_take
            print(f"  Processed {samples_processed}/{args.max_samples} samples...")
            
    print("Concatenating latents...")
    test_latents = torch.cat(test_latents, dim=0)
    
    latent_stats = {
        "mean": float(test_latents.mean().item()),
        "std": float(test_latents.std().item()),
        "min": float(test_latents.min().item()),
        "max": float(test_latents.max().item())
    }
    print(f"Latent stats - Mean: {latent_stats['mean']:.4f}, Std: {latent_stats['std']:.4f}, Min: {latent_stats['min']:.4f}, Max: {latent_stats['max']:.4f}")
    
    results["section_2"] = {
        "stability_passed": stability_passed,
        "latent_stats": latent_stats
    }
    
    # ==========================================
    # SECTION 3: Test Decoder-Only Reconstruction
    # ==========================================
    print("\n--- Running Section 3: Test Decoder-Only Reconstruction ---")
    decoder_only_preds = []
    context_targets_wf = []
    context_r_peaks_list = []
    
    test_targets_wf = []
    test_r_peaks = []
    
    samples_processed = 0
    with torch.no_grad():
        for batch in eval_loader:
            if samples_processed >= args.max_samples:
                break
            
            batch_context_wf = batch[0].to(device)
            batch_context_t = batch[1][0].to(device)
            batch_target_wf = batch[2]
            batch_segment_r_peaks = batch[4]
            
            # Batched CDE latent trajectory interpolation
            z_t_interpolated = get_interpolated_latent_trajectory(model.encoder, batch_context_wf, batch_context_t)
            
            # Batched decoding (passing context times to SIREN)
            pred_wf = model.decoder(z_t_interpolated, batch_context_t) # [batch, 1000, 12]
            
            batch_size = batch_context_wf.shape[0]
            remaining = args.max_samples - samples_processed
            actual_take = min(batch_size, remaining)
            
            pred_wf_cpu = pred_wf[:actual_take].cpu()
            
            for b in range(actual_take):
                idx = samples_processed + b
                decoder_only_preds.append(pred_wf_cpu[b:b+1])
                context_targets_wf.append(batch[0][b].unsqueeze(0))
                
                # Extract context R-peaks (needs to be sequential, which is fine since it's cheap)
                rec, start_idx = test_dataset.samples[idx]
                ann_samples = test_dataset.annotations[rec]
                context_start = start_idx
                context_end = start_idx + test_dataset.context_pts
                mask_context = (ann_samples >= context_start) & (ann_samples < context_end)
                context_r_peaks = ann_samples[mask_context] - context_start
                context_r_peaks_list.append(context_r_peaks)
                
                # Target future target variables
                test_targets_wf.append(batch_target_wf[b].unsqueeze(0))
                test_r_peaks.append(batch_segment_r_peaks[b])
                
                # Visual plots for first 20 examples
                if idx < 20:
                    pred_wf_np = pred_wf_cpu[b, :, 1].numpy()
                    context_wf_np = batch[0][b, :, 1].numpy()
                    peaks_np = context_r_peaks.cpu().numpy()
                    try:
                        _, info = nk.ecg_peaks(pred_wf_np, sampling_rate=100)
                        pred_peaks = info["ECG_R_Peaks"]
                    except Exception:
                        pred_peaks = np.array([])
                        
                    plot_and_save_ecg(
                        original=context_wf_np,
                        reconstructed=pred_wf_np,
                        original_peaks=peaks_np,
                        reconstructed_peaks=pred_peaks,
                        save_path=os.path.join(args.save_dir, "plots", f"decoder_only_recon_sample_{idx}.png"),
                        title=f"Decoder-Only Context Reconstruction (Sample {idx})"
                    )
            
            samples_processed += actual_take
            print(f"  Processed {samples_processed}/{args.max_samples} samples...")
            
    decoder_metrics = evaluate_waveform_predictions(decoder_only_preds, context_targets_wf, context_r_peaks_list)
    print("Decoder-only reconstruction metrics:")
    for k, v in decoder_metrics.items():
        print(f"  {k}: {v:.5f}")
    results["section_3"] = decoder_metrics
    
    # ==========================================
    # SECTION 4: Test Oracle Future Latent Reconstruction
    # ==========================================
    print("\n--- Running Section 4: Test Oracle Future Latent Reconstruction ---")
    oracle_preds = []
    
    samples_processed = 0
    with torch.no_grad():
        for batch in eval_loader:
            if samples_processed >= args.max_samples:
                break
                
            batch_target_wf = batch[2].to(device)
            batch_target_t = batch[3][0].to(device)
            
            # Batched CDE latent trajectory interpolation
            z_future_interpolated = get_interpolated_latent_trajectory(model.encoder, batch_target_wf, batch_target_t)
            
            # Batched decoding (passing future target times to SIREN)
            pred_wf = model.decoder(z_future_interpolated, batch_target_t)
            
            batch_size = batch_target_wf.shape[0]
            remaining = args.max_samples - samples_processed
            actual_take = min(batch_size, remaining)
            
            pred_wf_cpu = pred_wf[:actual_take].cpu()
            
            for b in range(actual_take):
                idx = samples_processed + b
                oracle_preds.append(pred_wf_cpu[b:b+1])
                
                # Visual plots for first 20 examples
                if idx < 20:
                    pred_wf_np = pred_wf_cpu[b, :, 1].numpy()
                    target_wf_np = batch[2][b, :, 1].numpy()
                    peaks_np = batch[4][b].cpu().numpy()
                    try:
                        _, info = nk.ecg_peaks(pred_wf_np, sampling_rate=100)
                        pred_peaks = info["ECG_R_Peaks"]
                    except Exception:
                        pred_peaks = np.array([])
                        
                    plot_and_save_ecg(
                        original=target_wf_np,
                        reconstructed=pred_wf_np,
                        original_peaks=peaks_np,
                        reconstructed_peaks=pred_peaks,
                        save_path=os.path.join(args.save_dir, "plots", f"oracle_recon_sample_{idx}.png"),
                        title=f"Oracle Future Reconstruction (Sample {idx})"
                    )
            
            samples_processed += actual_take
            print(f"  Processed {samples_processed}/{args.max_samples} samples...")
            
    oracle_metrics = evaluate_waveform_predictions(oracle_preds, test_targets_wf, test_r_peaks)
    print("Oracle reconstruction metrics:")
    for k, v in oracle_metrics.items():
        print(f"  {k}: {v:.5f}")
    results["section_4"] = oracle_metrics
    
    # ==========================================
    # SECTION 5: Build Waveform Baselines
    # ==========================================
    print("\n--- Running Section 5: Build Waveform Baselines ---")
    
    # 1. Zero Baseline
    zero_preds = [torch.zeros_like(twf) for twf in test_targets_wf[:args.max_samples]]
    zero_metrics = evaluate_waveform_predictions(zero_preds, test_targets_wf[:args.max_samples], test_r_peaks[:args.max_samples])
    print("Zero Baseline metrics:")
    for k, v in zero_metrics.items():
         print(f"  {k}: {v:.5f}")
         
    # 2. Mean Waveform Baseline
    print("Computing mean training waveform...")
    all_train_wf = []
    for i, batch in enumerate(train_loader):
        _, _, target_wf, _, _ = batch
        all_train_wf.append(target_wf)
        if len(all_train_wf) * target_wf.size(0) >= 1000:
            break
    mean_waveform = torch.cat(all_train_wf, dim=0).mean(dim=0, keepdim=True) # [1, 1000, 12]
    
    mean_wf_preds = [mean_waveform for _ in range(min(args.max_samples, len(test_dataset)))]
    mean_wf_metrics = evaluate_waveform_predictions(mean_wf_preds, test_targets_wf[:args.max_samples], test_r_peaks[:args.max_samples])
    print("Mean Waveform Baseline metrics:")
    for k, v in mean_wf_metrics.items():
         print(f"  {k}: {v:.5f}")
         
    # 3. Short-term Persistence Baseline
    persistence_preds = []
    for i in range(min(args.max_samples, len(test_dataset))):
         context_wf, _, _, _, _ = test_dataset[i]
         persistence_preds.append(context_wf.unsqueeze(0))
    persistence_metrics = evaluate_waveform_predictions(persistence_preds, test_targets_wf[:args.max_samples], test_r_peaks[:args.max_samples])
    print("Persistence Baseline metrics:")
    for k, v in persistence_metrics.items():
         print(f"  {k}: {v:.5f}")
         
    # 4. Random Guessing Baseline
    shuffled_indices = list(range(min(args.max_samples, len(test_dataset))))
    random.shuffle(shuffled_indices)
    random_preds = [test_targets_wf[idx] for idx in shuffled_indices]
    random_wf_metrics = evaluate_waveform_predictions(random_preds, test_targets_wf[:args.max_samples], test_r_peaks[:args.max_samples])
    print("Random Guessing Baseline metrics:")
    for k, v in random_wf_metrics.items():
         print(f"  {k}: {v:.5f}")
         
    results["section_5"] = {
        "zero": zero_metrics,
        "mean_waveform": mean_wf_metrics,
        "persistence": persistence_metrics,
        "random_guessing": random_wf_metrics
    }
    
    # ==========================================
    # SECTION 6: Build Latent Dynamics Baselines
    # ==========================================
    print("\n--- Running Section 6: Build Latent Dynamics Baselines ---")
    
    # Extract training and testing latents
    print("Extracting train latent pairs (z0, z_dt)...")
    z0_train, z_dt_train_dict = extract_latent_pairs(model.encoder, train_loader, device, max_samples=1000)
    print("Extracting test latent pairs (z0, z_dt)...")
    z0_test, z_dt_test_dict = extract_latent_pairs(model.encoder, test_loader, device, max_samples=args.max_samples)
    
    # Train Linear and MLP models
    print("Training Linear and MLP predictors on frozen latents...")
    linear_models, mlp_model = train_latent_baselines(z0_train, z_dt_train_dict, device, epochs=10)
    
    # Evaluate every latent baseline per time gap
    latent_baselines_results = {dt_val: {} for dt_val in range(1, 11)}
    
    linear_models.eval()
    mlp_model.eval()
    
    with torch.no_grad():
        for dt_val in range(1, 11):
            target_z = z_dt_test_dict[dt_val].to(device)
            z0_test_dev = z0_test.to(device)
            
            # 1. Mean baseline (mean train latent for this dt)
            mean_z = z_dt_train_dict[dt_val].mean(dim=0, keepdim=True).to(device).expand(target_z.size(0), -1)
            mean_metrics = compute_latent_metrics(mean_z, target_z, z0_test_dev)
            
            # 2. Persistence baseline (z_hat = z0)
            persist_metrics = compute_latent_metrics(z0_test_dev, target_z, z0_test_dev)
            
            # 3. Linear baseline
            linear_pred = linear_models[str(dt_val)](z0_test_dev)
            linear_metrics = compute_latent_metrics(linear_pred, target_z, z0_test_dev)
            
            # 4. MLP baseline
            dt_tensor = torch.full((z0_test_dev.size(0),), float(dt_val), device=device)
            mlp_pred = mlp_model(z0_test_dev, dt_tensor)
            mlp_metrics = compute_latent_metrics(mlp_pred, target_z, z0_test_dev)
            
            # 5. Random baseline (shuffle targets across batch)
            shuffled_idx = torch.randperm(target_z.size(0))
            random_z = target_z[shuffled_idx]
            random_metrics = compute_latent_metrics(random_z, target_z, z0_test_dev)
            
            latent_baselines_results[dt_val] = {
                "mean": mean_metrics,
                "persistence": persist_metrics,
                "linear": linear_metrics,
                "mlp": mlp_metrics,
                "random": random_metrics
            }
            
    print("\nLatent Dynamics Baselines results for dt = 10.0s:")
    for b_name, metrics in latent_baselines_results[10].items():
        print(f"  {b_name}: MSE: {metrics['mse']:.5f}, MAE: {metrics['mae']:.5f}, CosSim: {metrics['cosine_similarity']:.4f}, R2: {metrics['r2_vs_persistence']:.4f}")
        
    results["section_6"] = latent_baselines_results
    
    # Save the intermediate results to JSON file
    with open(os.path.join(args.save_dir, "verification_metrics_temp.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nIntermediate verification results written to {os.path.join(args.save_dir, 'verification_metrics_temp.json')}")
    save_json_metrics(results, args)
    
    # ==========================================
    # SECTION 7: Evaluate SDE Latent Prediction (requires SDE checkpoint)
    # ==========================================
    print("\n--- Checkpoint Check for Section 7: Evaluate SDE Latent Prediction ---")
    if not os.path.exists(args.checkpoint_path):
        print(f"\nCheckpoint not found at {args.checkpoint_path}!")
        print("PAUSING: Please run the training script to save the SDE model checkpoint under './checkpoints/neurosde_final.pt'.")
        print("After saving the checkpoint, send a proceed command to continue with Section 7 evaluation.")
        return
        
    print(f"\nCheckpoint found at {args.checkpoint_path}! Loading SDE checkpoint...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    sde_latent_results = {}
    with torch.no_grad():
        for dt_val in range(1, 11):
            target_z = z_dt_test_dict[dt_val].to(device)
            z0_test_dev = z0_test.to(device)
            
            # Predict future latent using SDE solver
            # query_times has size [1], representing offset dt_val
            query_time = torch.tensor([float(dt_val)], device=device)
            # solver returns [batch, 1, latent_dim]
            sde_pred = model.solver(z0_test_dev, query_time).squeeze(1)
            
            sde_metrics = compute_latent_metrics(sde_pred, target_z, z0_test_dev)
            sde_latent_results[dt_val] = sde_metrics
            
    print("\nSDE Latent Dynamics results for dt = 10.0s:")
    print(f"  SDE Solver: MSE: {sde_latent_results[10]['mse']:.5f}, MAE: {sde_latent_results[10]['mae']:.5f}, CosSim: {sde_latent_results[10]['cosine_similarity']:.4f}, R2: {sde_latent_results[10]['r2_vs_persistence']:.4f}")
    
    results["section_7"] = sde_latent_results
    
    # Save final results
    with open(os.path.join(args.save_dir, "verification_metrics_final.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"Final verification metrics written to {os.path.join(args.save_dir, 'verification_metrics_final.json')}")
    save_json_metrics(results, args)

if __name__ == "__main__":
    main()
