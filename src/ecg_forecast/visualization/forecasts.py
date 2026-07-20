import os
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def plot_lead2_forecast_panel(
    context_wf: np.ndarray,         # [500]
    gt_future_wf: np.ndarray,       # [200]
    posterior_recon: np.ndarray,    # [200]
    prior_samples: np.ndarray,      # [N_samples, 200]
    save_path: str,
    title_suffix: str = "",
):
    """Plots 4-panel figure for Lead II:
    1. Context waveform
    2. Ground-truth future waveform
    3. Posterior reconstruction
    4. Prior forecast samples + ensemble mean + 90% interval
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=False)

    ctx_times = np.linspace(-5.0, 0.0, len(context_wf))
    fut_times = np.linspace(0.0, 2.0, len(gt_future_wf))

    # Panel 1: Context Waveform
    axes[0].plot(ctx_times, context_wf, color="navy", lw=1.5, label="Context")
    axes[0].axvline(0.0, color="gray", linestyle="--")
    axes[0].set_ylabel("Lead II (mV)")
    axes[0].set_title(f"1. Context Waveform (5.0s) {title_suffix}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    # Panel 2: Ground-Truth Future Waveform
    axes[1].plot(fut_times, gt_future_wf, color="black", lw=1.5, label="Ground Truth Future")
    axes[1].axvline(0.0, color="gray", linestyle="--")
    axes[1].set_ylabel("Lead II (mV)")
    axes[1].set_title("2. Ground-Truth Future (2.0s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    # Panel 3: Posterior Reconstruction
    axes[2].plot(fut_times, gt_future_wf, color="gray", alpha=0.5, label="GT")
    axes[2].plot(fut_times, posterior_recon, color="crimson", lw=1.5, label="Posterior Recon")
    axes[2].axvline(0.0, color="gray", linestyle="--")
    axes[2].set_ylabel("Lead II (mV)")
    axes[2].set_title("3. Posterior Latent SDE Reconstruction")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    # Panel 4: Prior Forecast Samples
    ensemble_mean = np.mean(prior_samples, axis=0)
    p5 = np.percentile(prior_samples, 5, axis=0)
    p95 = np.percentile(prior_samples, 95, axis=0)

    for i in range(min(16, prior_samples.shape[0])):
        axes[3].plot(fut_times, prior_samples[i], color="lightsteelblue", alpha=0.4, lw=0.8)

    axes[3].plot(fut_times, gt_future_wf, color="black", linestyle=":", lw=1.2, label="GT")
    axes[3].plot(fut_times, ensemble_mean, color="royalblue", lw=1.8, label="Ensemble Mean")
    axes[3].fill_between(fut_times, p5, p95, color="royalblue", alpha=0.2, label="90% Interval")
    axes[3].axvline(0.0, color="gray", linestyle="--")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Lead II (mV)")
    axes[3].set_title("4. Prior Latent SDE Multi-Sample Forecast")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_12lead_forecast_page(
    context_wf: np.ndarray,      # [500, 12]
    gt_future_wf: np.ndarray,    # [200, 12]
    prior_mean: np.ndarray,      # [200, 12]
    save_path: str,
    lead_names: Optional[list] = None,
):
    """Plots 12-lead forecast comparison grid figure."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if lead_names is None:
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    fig, axes = plt.subplots(4, 3, figsize=(15, 10))
    axes = axes.flatten()

    ctx_times = np.linspace(-5.0, 0.0, context_wf.shape[0])
    fut_times = np.linspace(0.0, 2.0, gt_future_wf.shape[0])

    num_leads = min(12, context_wf.shape[1])
    for l in range(num_leads):
        ax = axes[l]
        name = lead_names[l] if l < len(lead_names) else f"Lead {l+1}"

        ax.plot(ctx_times, context_wf[:, l], color="navy", lw=1.0, alpha=0.7)
        ax.plot(fut_times, gt_future_wf[:, l], color="black", lw=1.2, label="GT")
        ax.plot(fut_times, prior_mean[:, l], color="crimson", lw=1.2, label="Forecast")
        ax.axvline(0.0, color="gray", linestyle="--")
        ax.set_title(f"Lead {name}")
        ax.grid(True, alpha=0.3)
        if l == 0:
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
