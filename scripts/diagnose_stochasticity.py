import argparse
import os
import torch
from torch.utils.data import DataLoader
import torchsde

from ecg_forecast.config import load_config
from ecg_forecast.data.windows import SignatureECGDataset
from ecg_forecast.data.collate import ecg_collate_fn
from ecg_forecast.models.cnsde import ConditionalNeuralSDE
from ecg_forecast.signatures.signature import get_signature_dim


def run_stochasticity_diagnostics(
    model: ConditionalNeuralSDE,
    loader: DataLoader,
    num_samples: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model.eval()
    model.to(device)

    # Grab a representative batch
    batch = next(iter(loader))
    sig_x = batch["context_signature"].to(device)
    y0 = batch["context_waveform"][:, -1:, :].to(device)

    # Use first N contexts to keep memory reasonable
    max_b = min(16, sig_x.shape[0])
    sig_x = sig_x[:max_b]
    y0 = y0[:max_b]
    B = max_b
    K = num_samples
    D = model.latent_dim
    noise_dim = model.initial_noise_dim

    print(f"\nRunning Section 15 Stochasticity Diagnostics on B={B} contexts with K={K} samples...")

    # --- A. Initial-noise variation only (Fixed Brownian path, variable epsilon) ---
    # Fix 1 Brownian motion for each batch item and replicate it across all K samples
    bm_fixed = torchsde.BrownianInterval(
        t0=0.0,
        t1=model.t_end,
        size=(B, D),
        device=device,
        levy_area_approximation="none",
    )
    # Generate K samples one-by-one with identical bm_fixed
    samples_a = []
    latents_a = []
    with torch.no_grad():
        for k in range(K):
            eps_k = torch.randn(B, noise_dim, device=device)
            # Forward with 1 sample and fixed BM
            wf_k, lat_k = model(
                sig_x,
                y0,
                num_samples=1,
                epsilon=eps_k,
                bm=bm_fixed,
                use_adjoint=False,
            )
            samples_a.append(wf_k[:, 0])   # [B, 200, C]
            latents_a.append(lat_k[:, 0])  # [B, 201, D]

    wf_a = torch.stack(samples_a, dim=1)   # [B, K, 200, C]
    lat_a = torch.stack(latents_a, dim=1)  # [B, K, 201, D]
    init_wf_std = wf_a.std(dim=1).mean().item()
    init_lat_std = lat_a.std(dim=1).mean().item()

    # --- B. Brownian variation only (Fixed epsilon / z0, variable Brownian path) ---
    fixed_eps = torch.randn(B, noise_dim, device=device)
    samples_b = []
    latents_b = []
    with torch.no_grad():
        for k in range(K):
            # Fresh Brownian motion for each k
            bm_k = torchsde.BrownianInterval(
                t0=0.0,
                t1=model.t_end,
                size=(B, D),
                device=device,
                levy_area_approximation="none",
            )
            wf_k, lat_k = model(
                sig_x,
                y0,
                num_samples=1,
                epsilon=fixed_eps,
                bm=bm_k,
                use_adjoint=False,
            )
            samples_b.append(wf_k[:, 0])
            latents_b.append(lat_k[:, 0])

    wf_b = torch.stack(samples_b, dim=1)
    lat_b = torch.stack(latents_b, dim=1)
    brownian_wf_std = wf_b.std(dim=1).mean().item()
    brownian_lat_std = lat_b.std(dim=1).mean().item()

    # --- C. Combined variation (Both epsilon and Brownian motion vary) ---
    with torch.no_grad():
        wf_c, lat_c = model(
            sig_x,
            y0,
            num_samples=K,
            use_adjoint=False,
        )
    combined_wf_std = wf_c.std(dim=1).mean().item()
    combined_lat_std = lat_c.std(dim=1).mean().item()

    # --- Drift and Diffusion Norms ---
    # Evaluate drift and diffusion at midpoint along the combined trajectories
    c = model.context_encoder(sig_x)
    model.sde_func.set_context(c.unsqueeze(1).expand(B, K, -1).contiguous().view(B * K, -1))
    mid_idx = lat_c.shape[2] // 2
    z_mid = lat_c[:, :, mid_idx, :].contiguous().view(B * K, D)
    t_mid = model.ts[mid_idx]

    with torch.no_grad():
        f_mid = model.sde_func.f(t_mid, z_mid)
        g_mid = model.sde_func.g(t_mid, z_mid)
        mean_drift_norm = f_mid.norm(dim=-1).mean().item()
        mean_diff_norm = g_mid.norm(dim=-1).mean().item()

    results = {
        "initial_noise_waveform_std": init_wf_std,
        "brownian_waveform_std": brownian_wf_std,
        "combined_waveform_std": combined_wf_std,
        "initial_noise_latent_std": init_lat_std,
        "brownian_latent_std": brownian_lat_std,
        "combined_latent_std": combined_lat_std,
        "mean_drift_norm": mean_drift_norm,
        "mean_diffusion_norm": mean_diff_norm,
    }

    print("\n--- Diagnostic Results ---")
    print(f"initial_noise_waveform_std: {init_wf_std:.5f}")
    print(f"brownian_waveform_std:      {brownian_wf_std:.5f}")
    print(f"combined_waveform_std:      {combined_wf_std:.5f}")
    print(f"initial_noise_latent_std:   {init_lat_std:.5f}")
    print(f"brownian_latent_std:        {brownian_lat_std:.5f}")
    print(f"combined_latent_std:        {combined_lat_std:.5f}")
    print(f"E[||f(t, z, c)||]:          {mean_drift_norm:.5f}")
    print(f"E[||g(t, z, c)||]:          {mean_diff_norm:.5f}")
    print("--------------------------\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Section 15 Stochasticity Diagnostics")
    parser.add_argument("checkpoint_pos", nargs="?", default=None, help="Path to checkpoint (optional, positional)")
    parser.add_argument("--checkpoint", "-c", dest="checkpoint_flag", type=str, default=None, help="Path to checkpoint (optional)")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (optional)")
    parser.add_argument("--num_samples", "-k", type=int, default=32, help="Number of diagnostic samples K")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_flag or args.checkpoint_pos
    if not checkpoint_path:
        default_ckpt = "checkpoints/lead2_cnsde/best_cnsde.pt"
        if os.path.exists(default_ckpt):
            checkpoint_path = default_ckpt

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = None
    ckpt_config = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict) and "config" in ckpt:
            ckpt_config = ckpt["config"]
    else:
        print("Evaluating randomly initialized model (no checkpoint found/supplied)")

    if args.config:
        cfg = load_config(args.config)
    elif ckpt_config is not None:
        cfg = load_config(ckpt_config)
    else:
        cfg = load_config("configs/lead2_cnsde.yaml")

    sig_dir = getattr(cfg.signature, "signatures_dir", "artifacts/signatures")
    dataset = SignatureECGDataset(config=cfg.data, split="val", signatures_dir=sig_dir)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=ecg_collate_fn)

    sig_dim = get_signature_dim(
        input_channels=cfg.model.num_leads,
        depth=cfg.signature.depth,
        dyadic_depth=cfg.signature.dyadic_depth,
        lead_lag=cfg.signature.lead_lag,
    )
    model = ConditionalNeuralSDE.from_config(cfg.model, cfg.sde, sig_dim=sig_dim)

    if ckpt is not None:
        state_dict = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
        model.load_state_dict(state_dict)
        print("Loaded model weights from checkpoint successfully.")

    run_stochasticity_diagnostics(model, loader, num_samples=args.num_samples, device=device)


if __name__ == "__main__":
    main()
