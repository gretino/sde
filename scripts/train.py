import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ecg_forecast.config import load_config
from ecg_forecast.data.windows import SignatureECGDataset
from ecg_forecast.data.collate import ecg_collate_fn
from ecg_forecast.models.cnsde import ConditionalNeuralSDE
from ecg_forecast.losses.csig import ConditionalSignatureLoss
from ecg_forecast.metrics.waveform import compute_cnsde_sample_metrics
from ecg_forecast.metrics.rhythm import compute_cnsde_rhythm_metrics
from ecg_forecast.signatures.signature import get_signature_dim
from precompute_signatures import compute_and_save_signatures


def compute_module_grad_norm(module: nn.Module) -> float:
    total_norm = 0.0
    for p in module.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def train_one_epoch(
    model: ConditionalNeuralSDE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: ConditionalSignatureLoss,
    grad_clip: float,
    num_samples: int,
    device: str,
):
    model.train()
    total_loss = 0.0
    num_batches = 0

    grad_norms = {
        "context_encoder": 0.0,
        "initial_state": 0.0,
        "drift": 0.0,
        "diffusion": 0.0,
        "readout": 0.0,
    }

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        sig_x = batch["context_signature"].to(device)
        target_sig_y = batch["conditional_future_signature"].to(device)
        y0 = batch["context_waveform"][:, -1:, :].to(device)

        optimizer.zero_grad()

        waveform_samples, latent_samples = model(
            sig_x,
            y0,
            num_samples=num_samples,
            use_adjoint=True,
        )

        loss = loss_fn(waveform_samples, target_sig_y)
        loss.backward()

        # Log submodule gradient norms
        ce_norm = compute_module_grad_norm(model.context_encoder)
        is_norm = compute_module_grad_norm(model.initial_state_net)
        dr_norm = compute_module_grad_norm(model.sde_func.drift_net)
        di_norm = compute_module_grad_norm(model.sde_func.diffusion_net)
        ro_norm = compute_module_grad_norm(model.readout)

        grad_norms["context_encoder"] += ce_norm
        grad_norms["initial_state"] += is_norm
        grad_norms["drift"] += dr_norm
        grad_norms["diffusion"] += di_norm
        grad_norms["readout"] += ro_norm

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "diff_gnorm": f"{di_norm:.4e}"})

    avg_loss = total_loss / max(1, num_batches)
    avg_grad_norms = {k: v / max(1, num_batches) for k, v in grad_norms.items()}
    return avg_loss, avg_grad_norms


def evaluate(
    model: ConditionalNeuralSDE,
    loader: DataLoader,
    loss_fn: ConditionalSignatureLoss,
    num_samples: int,
    device: str,
):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            sig_x = batch["context_signature"].to(device)
            target_sig_y = batch["conditional_future_signature"].to(device)
            future_wf = batch["future_waveform"].to(device)
            y0 = batch["context_waveform"][:, -1:, :].to(device)

            waveform_samples, latent_samples = model(
                sig_x,
                y0,
                num_samples=num_samples,
                use_adjoint=False,
            )

            loss = loss_fn(waveform_samples, target_sig_y)
            total_loss += loss.item()
            num_batches += 1

            batch_metrics = compute_cnsde_sample_metrics(
                waveform_samples=waveform_samples,
                ground_truth=future_wf,
                latent_samples=latent_samples,
            )
            rhythm_metrics = compute_cnsde_rhythm_metrics(
                waveform_samples=waveform_samples,
                target_r_peaks_list=batch.get("future_r_peaks"),
                ground_truth=future_wf,
                sampling_rate=100,
            )
            batch_metrics.update(rhythm_metrics)
            all_metrics.append(batch_metrics)

    avg_loss = total_loss / max(1, num_batches)
    avg_metrics = {}
    if all_metrics:
        for k in all_metrics[0].keys():
            avg_metrics[k] = float(sum(m[k] for m in all_metrics) / len(all_metrics))

    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Conditional Neural SDE for ECG Forecasting")
    parser.add_argument("config_pos", nargs="?", default=None, help="Path to config YAML (positional)")
    parser.add_argument("--config", "-c", dest="config_flag", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--epochs", "--max-epoch", "--max_epoch", dest="epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--num_samples", "--monte_carlo_samples", type=int, default=None, help="Override training MC samples K")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    args = parser.parse_args()

    config_path = args.config_flag or args.config_pos or "configs/lead2_cnsde.yaml"
    cfg = load_config(config_path)

    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.num_samples is not None:
        cfg.training.num_samples = args.num_samples
    if args.lr is not None:
        cfg.training.learning_rate = args.lr

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.training.seed)

    # Ensure precomputed signature artifacts exist according to config
    sig_dir = getattr(cfg.signature, "signatures_dir", "artifacts/signatures")
    train_sig_file = os.path.join(sig_dir, "train_signatures.pt")
    val_sig_file = os.path.join(sig_dir, "val_signatures.pt")
    if not (os.path.exists(train_sig_file) and os.path.exists(val_sig_file)):
        print(f"Signatures not found in {sig_dir}. Precomputing signatures from config...")
        compute_and_save_signatures(cfg, output_dir=sig_dir, device=device)

    print(f"Loading datasets with config: {config_path}")
    train_dataset = SignatureECGDataset(config=cfg.data, split="train", signatures_dir=sig_dir)
    val_dataset = SignatureECGDataset(config=cfg.data, split="val", signatures_dir=sig_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=ecg_collate_fn,
        num_workers=cfg.training.num_workers,
        pin_memory=(device == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=ecg_collate_fn,
        num_workers=cfg.training.num_workers,
        pin_memory=(device == "cuda"),
    )

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    sig_dim = get_signature_dim(
        input_channels=cfg.model.num_leads,
        depth=cfg.signature.depth,
        dyadic_depth=cfg.signature.dyadic_depth,
        lead_lag=cfg.signature.lead_lag,
    )

    model = ConditionalNeuralSDE.from_config(cfg.model, cfg.sde, sig_dim=sig_dim).to(device)

    # Load normalization statistics for loss
    sigy_mean_path = os.path.join(sig_dir, "sigy_mean.pt")
    sigy_std_path = os.path.join(sig_dir, "sigy_std.pt")
    sigy_mean = torch.load(sigy_mean_path, map_location=device, weights_only=False) if os.path.exists(sigy_mean_path) else None
    sigy_std = torch.load(sigy_std_path, map_location=device, weights_only=False) if os.path.exists(sigy_std_path) else None

    loss_fn = ConditionalSignatureLoss(
        depth=cfg.signature.depth,
        dyadic_depth=cfg.signature.dyadic_depth,
        lead_lag=cfg.signature.lead_lag,
        sigy_mean=sigy_mean,
        sigy_std=sigy_std,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    use_wandb = cfg.training.use_wandb and not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project=cfg.training.wandb_project, name=cfg.training.run_name, config=vars(cfg))
        except Exception as e:
            print(f"WandB init failed: {e}. Proceeding without WandB.")
            use_wandb = False

    print(f"Beginning training on device {device} for {cfg.training.epochs} epochs...")
    for epoch in range(1, cfg.training.epochs + 1):
        t0 = time.time()
        train_loss, grad_norms = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            grad_clip=cfg.training.grad_clip,
            num_samples=cfg.training.num_samples,
            device=device,
        )

        val_loss, val_metrics = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            num_samples=cfg.validation.num_samples,
            device=device,
        )
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d} | Train CSig: {train_loss:.5f} | Val CSig: {val_loss:.5f} | "
            f"Val Pearson: {val_metrics.get('median_pearson', 0):.3f} (best: {val_metrics.get('best_of_k_pearson', 0):.3f}) | "
            f"WF Std: {val_metrics.get('waveform_sample_std_mean', 0):.4f} | "
            f"Diff GNorm: {grad_norms['diffusion']:.2e} | Time: {elapsed:.1f}s"
        )

        if use_wandb:
            import wandb
            log_dict = {
                "train/csig_loss": train_loss,
                "val/csig_loss": val_loss,
                **{f"val/{k}": v for k, v in val_metrics.items()},
                **{f"grad_norm/{k}": v for k, v in grad_norms.items()},
                "epoch": epoch,
            }
            wandb.log(log_dict)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(cfg.training.checkpoint_dir, "best_cnsde.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"  --> Saved new best checkpoint to {ckpt_path} (Val CSig: {val_loss:.5f})")


if __name__ == "__main__":
    main()
