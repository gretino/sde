import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from ecg_forecast.config import load_config, Config
from ecg_forecast.data.windows import SignatureECGDataset
from ecg_forecast.data.collate import ecg_collate_fn
from ecg_forecast.models.cnsde import ConditionalNeuralSDE
from ecg_forecast.signatures.signature import get_signature_dim
from ecg_forecast.metrics.waveform import compute_cnsde_sample_metrics
from ecg_forecast.metrics.rhythm import compute_cnsde_rhythm_metrics, detect_r_peaks_lead


def plot_forecast_panel(
    model: ConditionalNeuralSDE,
    loader: DataLoader,
    num_samples: int = 16,
    num_plots: int = 4,
    denormalize: bool = False,
    output_path: str = "artifacts/visualizations/cnsde_forecast_panel.png",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    title_extra: str = "",
):
    """Generates a publication-grade two-column visualization panel:
    - Left Column: Full 7-second context & forecast overview (-5.0s to +2.0s)
    - Right Column: Zoomed-in 2-second future window (0.0s to 2.0s) with stochastic trajectories,
      confidence ribbon, detected peaks, and diversity metrics.
    """
    model.eval()
    model.to(device)

    batch = next(iter(loader))
    sig_x = batch["context_signature"].to(device)
    y0 = batch["context_waveform"][:, -1:, :].to(device)

    B = min(num_plots, sig_x.shape[0])
    sig_x = sig_x[:B]
    y0 = y0[:B]

    ctx_wf = batch["context_waveform"][:B].cpu().numpy()       # [B, 500, C]
    fut_wf = batch["future_waveform"][:B].cpu().numpy()        # [B, 200, C]
    fut_wf_torch = batch["future_waveform"][:B]
    record_ids = batch.get("record_ids", [f"Rec_{i}" for i in range(B)])[:B]

    mu = batch["normalization_mean"][:B].cpu().numpy()         # [B, 1]
    sigma = batch["normalization_std"][:B].cpu().numpy()       # [B, 1]

    with torch.no_grad():
        wf_samples, lat_samples = model(
            sig_x,
            y0,
            num_samples=num_samples,
            use_adjoint=False,
        )

    samples_np = wf_samples.cpu().numpy()  # [B, K, 200, C]

    # Metrics
    sample_metrics = compute_cnsde_sample_metrics(wf_samples, fut_wf_torch, lat_samples)
    rhythm_metrics = compute_cnsde_rhythm_metrics(wf_samples, ground_truth=fut_wf_torch)

    print("\n" + "=" * 55)
    print("        ECG FORECAST EVALUATION SUMMARY")
    print("=" * 55)
    print(f"Sample Diversity Std (mean): {sample_metrics.get('waveform_sample_std_mean', 0.0):.4f}")
    print(f"Sample Diversity Std (max):  {sample_metrics.get('waveform_sample_std_max', 0.0):.4f}")
    print(f"Median Pearson Correlation:  {sample_metrics.get('median_pearson', 0.0):.3f}")
    print(f"Best-of-K Pearson:           {sample_metrics.get('best_of_k_pearson', 0.0):.3f}")
    print(f"Median MSE:                  {sample_metrics.get('median_mse', 0.0):.4f}")
    print(f"Best-of-K MSE:               {sample_metrics.get('best_of_k_mse', 0.0):.4f}")
    print(f"Zero R-peak Sample Fraction: {rhythm_metrics.get('zero_rpeak_pct', 0.0):.1f}%")
    print(f"Sample Estimated Heart Rate: {rhythm_metrics.get('sample_hr_mean', 0.0):.1f} BPM")
    print("=" * 55 + "\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 2-column figure layout: Left = Overview (Context+Future), Right = Zoomed Future
    fig, axes = plt.subplots(B, 2, figsize=(18, 3.6 * B), gridspec_kw={"width_ratios": [1.1, 1.0]})
    if B == 1:
        axes = np.expand_dims(axes, axis=0)

    fs = 100.0
    # Context time: -5.00s to 0.00s (500 steps)
    t_ctx = np.linspace(-5.0, 0.0, 500)
    # Future time: 0.00s to 2.00s (201 steps, anchored at t=0)
    t_fut = np.linspace(0.0, 2.0, 201)

    y_unit = "mV" if denormalize else "Norm. Amp."

    for b in range(B):
        ax_full = axes[b, 0]
        ax_zoom = axes[b, 1]

        scale = sigma[b, 0] if denormalize else 1.0
        shift = mu[b, 0] if denormalize else 0.0

        ctx_b = ctx_wf[b, :, 0] * scale + shift
        fut_b = fut_wf[b, :, 0] * scale + shift
        c_anchor = ctx_b[-1]

        samples_b = samples_np[b, :, :, 0] * scale + shift   # [K, 200]
        samples_anchored = np.concatenate([np.full((num_samples, 1), c_anchor), samples_b], axis=1) # [K, 201]

        fut_anchored = np.append([c_anchor], fut_b)         # [201]
        mean_anchored = np.mean(samples_anchored, axis=0)    # [201]
        p10 = np.percentile(samples_anchored, 10, axis=0)
        p90 = np.percentile(samples_anchored, 90, axis=0)
        p25 = np.percentile(samples_anchored, 25, axis=0)
        p75 = np.percentile(samples_anchored, 75, axis=0)

        rec_id = record_ids[b]

        # -------------------------------------------------------------
        # Left Panel: Full Context & Forecast Continuation
        # -------------------------------------------------------------
        # Context
        ax_full.plot(t_ctx, ctx_b, color="#1e293b", lw=1.5, label="Context (5s)" if b == 0 else None, zorder=4)
        # True future
        ax_full.plot(t_fut, fut_anchored, color="#10b981", lw=1.8, linestyle="--", label="True Future (2s)" if b == 0 else None, zorder=5)
        # Forecast ensemble mean & 10-90% ribbon
        ax_full.plot(t_fut, mean_anchored, color="#ef4444", lw=1.9, label="Forecast Mean" if b == 0 else None, zorder=6)
        ax_full.fill_between(t_fut, p10, p90, color="#3b82f6", alpha=0.25, label="10-90% Range" if b == 0 else None, zorder=2)
        # Context boundary line
        ax_full.axvline(x=0.0, color="#64748b", linestyle=":", lw=1.5, zorder=3)

        ax_full.set_ylabel(f"Lead II ({y_unit})", fontsize=10)
        ax_full.set_title(f"Window {b + 1} ({rec_id}) — Full View (-5s to +2s)", fontsize=11, fontweight="bold", loc="left")
        ax_full.grid(True, alpha=0.3, linestyle="--")
        if b == 0:
            ax_full.legend(loc="upper left", framealpha=0.9, fontsize=9)

        # -------------------------------------------------------------
        # Right Panel: Zoomed-in Future Horizon (0.0s to 2.0s)
        # -------------------------------------------------------------
        # 10-90% and 25-75% confidence ribbons
        ax_zoom.fill_between(t_fut, p10, p90, color="#93c5fd", alpha=0.30, label="10th-90th Percentile" if b == 0 else None, zorder=1)
        ax_zoom.fill_between(t_fut, p25, p75, color="#60a5fa", alpha=0.40, label="25th-75th Percentile" if b == 0 else None, zorder=2)

        # Individual stochastic sample trajectories
        cmap = plt.get_cmap("tab20")
        for k in range(min(num_samples, 16)):
            ax_zoom.plot(
                t_fut,
                samples_anchored[k],
                color=cmap(k % 20),
                alpha=0.55,
                lw=1.1,
                label=f"Sample Trajectories (K={num_samples})" if (b == 0 and k == 0) else None,
                zorder=3,
            )

        # Ensemble Mean
        ax_zoom.plot(t_fut, mean_anchored, color="#b91c1c", lw=2.2, label="Ensemble Mean" if b == 0 else None, zorder=6)

        # Ground truth future
        ax_zoom.plot(t_fut, fut_anchored, color="#047857", lw=2.0, linestyle="--", label="Ground Truth" if b == 0 else None, zorder=5)

        # Mark ground truth R-peaks if detected
        gt_peaks = detect_r_peaks_lead(fut_b, sampling_rate=int(fs))
        if len(gt_peaks) > 0:
            gt_peak_times = t_fut[gt_peaks + 1]
            gt_peak_vals = fut_anchored[gt_peaks + 1]
            ax_zoom.scatter(gt_peak_times, gt_peak_vals, color="#047857", s=36, marker="o", zorder=7, label="True R-peaks" if b == 0 else None)

        # Anchor divider at t=0
        ax_zoom.axvline(x=0.0, color="#64748b", linestyle=":", lw=1.5, zorder=4)

        # Window metrics
        b_std = float(np.std(samples_b, axis=0).mean())
        corrs = []
        for k in range(num_samples):
            s_k = samples_b[k]
            if np.std(s_k) > 1e-4 and np.std(fut_b) > 1e-4:
                r = float(np.corrcoef(s_k, fut_b)[0, 1])
                corrs.append(r if not np.isnan(r) else 0.0)
            else:
                corrs.append(0.0)
        best_r = max(corrs) if corrs else 0.0
        med_r = float(np.median(corrs)) if corrs else 0.0
        mse = float(np.mean((mean_anchored[1:] - fut_b) ** 2))

        ax_zoom.set_ylabel(f"Lead II ({y_unit})", fontsize=10)
        ax_zoom.set_title(
            f"Zoomed Future: Std_k={b_std:.3f} | Best r={best_r:.2f} (med: {med_r:.2f}) | MSE={mse:.3f}",
            fontsize=11,
            fontweight="bold",
            loc="left",
        )
        ax_zoom.set_xlim(-0.02, 2.02)
        ax_zoom.grid(True, alpha=0.35, linestyle="--")
        if b == 0:
            ax_zoom.legend(loc="upper right", framealpha=0.9, fontsize=8, ncol=2)

    axes[-1, 0].set_xlabel("Time (seconds relative to context end t=0)", fontsize=11)
    axes[-1, 1].set_xlabel("Future Time (seconds)", fontsize=11)

    title_main = f"Conditional Neural SDE ECG Forecasting — Multi-Sample Evaluation"
    if title_extra:
        title_main += f" ({title_extra})"
    fig.suptitle(title_main, fontsize=14, fontweight="bold", y=0.995)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved detailed forecast panel to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Conditional Neural SDE Forecasts from Checkpoint")
    parser.add_argument("checkpoint_pos", nargs="?", default=None, help="Path to model checkpoint (.pt) (positional)")
    parser.add_argument("--checkpoint", "-c", "--pt", dest="checkpoint_flag", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (optional; defaults to checkpoint config)")
    parser.add_argument("--output", "-o", type=str, default="artifacts/visualizations/cnsde_forecast_panel.png", help="Output image file")
    parser.add_argument("--num_samples", "-k", type=int, default=16, help="Monte Carlo samples per context (default: 16)")
    parser.add_argument("--num_plots", "-n", type=int, default=4, help="Number of windows to plot (default: 4)")
    parser.add_argument("--denormalize", "--raw", "--mv", action="store_true", help="Plot in physical units (mV) instead of normalized amplitude")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_flag or args.checkpoint_pos
    if not checkpoint_path:
        default_ckpt = "checkpoints/lead2_cnsde/best_cnsde.pt"
        if os.path.exists(default_ckpt):
            checkpoint_path = default_ckpt
            print(f"No checkpoint specified. Using default: {checkpoint_path}")
        else:
            raise FileNotFoundError(
                "No checkpoint specified and default 'checkpoints/lead2_cnsde/best_cnsde.pt' not found. "
                "Please provide a checkpoint path."
            )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint safely
    ckpt = None
    ckpt_config = None
    title_extra = ""
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(ckpt, dict) and "config" in ckpt:
            ckpt_config = ckpt["config"]
        if isinstance(ckpt, dict) and "epoch" in ckpt and "val_loss" in ckpt:
            title_extra = f"Epoch {ckpt['epoch']} | Val CSig: {ckpt['val_loss']:.4f}"
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Determine configuration
    if args.config:
        cfg = load_config(args.config)
        print(f"Using configuration from file: {args.config}")
    elif ckpt_config is not None:
        cfg = load_config(ckpt_config)
        print("Using configuration embedded in checkpoint")
    else:
        cfg = load_config("configs/lead2_cnsde.yaml")
        print("Using default configuration: configs/lead2_cnsde.yaml")

    sig_dir = getattr(cfg.signature, "signatures_dir", "artifacts/signatures")
    dataset = SignatureECGDataset(config=cfg.data, split="val", signatures_dir=sig_dir)
    loader = DataLoader(dataset, batch_size=args.num_plots, shuffle=False, collate_fn=ecg_collate_fn)

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
        print("Model weights loaded successfully.")

    plot_forecast_panel(
        model=model,
        loader=loader,
        num_samples=args.num_samples,
        num_plots=args.num_plots,
        denormalize=args.denormalize,
        output_path=args.output,
        device=device,
        title_extra=title_extra,
    )


if __name__ == "__main__":
    main()
