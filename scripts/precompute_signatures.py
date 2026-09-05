import argparse
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ecg_forecast.config import load_config
from ecg_forecast.data.windows import ECGWindowDataset
from ecg_forecast.data.collate import ecg_collate_fn
from ecg_forecast.signatures.signature import compute_signature_features


def compute_dataset_signatures(
    dataset: ECGWindowDataset,
    batch_size: int = 64,
    depth: int = 4,
    dyadic_depth: int = 2,
    lead_lag: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ecg_collate_fn,
        num_workers=0,
    )

    all_sig_x = []
    all_sig_y = []

    for batch in tqdm(loader, desc="Computing signatures"):
        ctx = batch["context_waveform"].to(device)  # [B, 500, 1]
        fut = batch["future_waveform"].to(device)   # [B, 200, 1]

        with torch.no_grad():
            sig_x = compute_signature_features(
                ctx,
                depth=depth,
                dyadic_depth=dyadic_depth,
                lead_lag=lead_lag,
            )
            sig_y = compute_signature_features(
                fut,
                depth=depth,
                dyadic_depth=dyadic_depth,
                lead_lag=lead_lag,
            )

        all_sig_x.append(sig_x.cpu())
        all_sig_y.append(sig_y.cpu())

    return torch.cat(all_sig_x, dim=0), torch.cat(all_sig_y, dim=0)


def fit_ridge_regression(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    alpha: float = 0.1,
):
    """Fits Ridge regression using double precision: Y = X @ W.T + b."""
    from sklearn.linear_model import Ridge
    X_np = X_train.detach().cpu().numpy().astype(np.float64)
    Y_np = Y_train.detach().cpu().numpy().astype(np.float64)

    reg = Ridge(alpha=alpha, fit_intercept=True)
    reg.fit(X_np, Y_np)

    weight = torch.from_numpy(reg.coef_).float()      # [D_y, D_x]
    bias = torch.from_numpy(reg.intercept_).float()   # [D_y]

    return weight, bias


def compute_and_save_signatures(
    cfg,
    output_dir: str = "artifacts/signatures",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 64,
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading datasets to precompute signatures...")
    train_dataset = ECGWindowDataset(config=cfg.data, split="train")
    val_dataset = ECGWindowDataset(config=cfg.data, split="val")

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print("Computing unnormalized signatures for training split...")
    raw_sigx_train, raw_sigy_train = compute_dataset_signatures(
        train_dataset,
        batch_size=batch_size,
        depth=cfg.signature.depth,
        dyadic_depth=cfg.signature.dyadic_depth,
        lead_lag=cfg.signature.lead_lag,
        device=device,
    )

    print("Computing unnormalized signatures for validation split...")
    raw_sigx_val, raw_sigy_val = compute_dataset_signatures(
        val_dataset,
        batch_size=batch_size,
        depth=cfg.signature.depth,
        dyadic_depth=cfg.signature.dyadic_depth,
        lead_lag=cfg.signature.lead_lag,
        device=device,
    )

    # Compute normalization statistics strictly from training split
    sigx_mean = raw_sigx_train.mean(dim=0)
    sigx_std = raw_sigx_train.std(dim=0).clamp_min(1e-5)

    sigy_mean = raw_sigy_train.mean(dim=0)
    sigy_std = raw_sigy_train.std(dim=0).clamp_min(1e-5)

    # Save normalization artifacts
    torch.save(sigx_mean, os.path.join(output_dir, "sigx_mean.pt"))
    torch.save(sigx_std, os.path.join(output_dir, "sigx_std.pt"))
    torch.save(sigy_mean, os.path.join(output_dir, "sigy_mean.pt"))
    torch.save(sigy_std, os.path.join(output_dir, "sigy_std.pt"))
    print("Saved normalization artifacts (sigx_mean, sigx_std, sigy_mean, sigy_std)")

    # Normalize features
    norm_sigx_train = (raw_sigx_train - sigx_mean) / sigx_std
    norm_sigy_train = (raw_sigy_train - sigy_mean) / sigy_std

    norm_sigx_val = (raw_sigx_val - sigx_mean) / sigx_std
    norm_sigy_val = (raw_sigy_val - sigy_mean) / sigy_std

    # Fit Ridge Regression: S(X) -> S(Y)
    print(f"Fitting Ridge Regression with alpha={cfg.signature.ridge_alpha}...")
    weight, bias = fit_ridge_regression(norm_sigx_train, norm_sigy_train, alpha=cfg.signature.ridge_alpha)

    ridge_dict = {"weight": weight, "bias": bias}
    torch.save(ridge_dict, os.path.join(output_dir, "ridge_model.pt"))
    with open(os.path.join(output_dir, "ridge_model.pkl"), "wb") as f:
        pickle.dump(ridge_dict, f)
    print("Saved ridge model artifacts")

    # Compute conditional future signature targets: S*_future(X) = X @ W.T + b
    cond_target_train = norm_sigx_train @ weight.T + bias
    cond_target_val = norm_sigx_val @ weight.T + bias

    # Save precomputed signature splits
    train_out = {
        "context_signature": norm_sigx_train,
        "conditional_future_signature": cond_target_train,
        "true_future_signature": norm_sigy_train,
    }
    torch.save(train_out, os.path.join(output_dir, "train_signatures.pt"))

    val_out = {
        "context_signature": norm_sigx_val,
        "conditional_future_signature": cond_target_val,
        "true_future_signature": norm_sigy_val,
    }
    torch.save(val_out, os.path.join(output_dir, "val_signatures.pt"))
    print("Saved precomputed train_signatures.pt and val_signatures.pt successfully!")


def main():
    parser = argparse.ArgumentParser(description="Precompute path signatures and fit conditional ridge target")
    parser.add_argument("config_pos", nargs="?", default=None, help="Path to config YAML (positional)")
    parser.add_argument("--config", "-c", dest="config_flag", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for signatures")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    config_path = args.config_flag or args.config_pos or "configs/lead2_cnsde.yaml"
    cfg = load_config(config_path)
    output_dir = args.output_dir or cfg.signature.signatures_dir or "artifacts/signatures"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    compute_and_save_signatures(
        cfg=cfg,
        output_dir=output_dir,
        device=device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
