import warnings
from typing import Dict, Optional
import numpy as np
import torch


def compute_waveform_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Computes MSE, MAE, and Pearson correlation coefficient between predicted and target waveforms safely."""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    if pred_np.size == 0 or target_np.size == 0:
        return {"mse": 0.0, "mae": 0.0, "pearson": 0.0}

    mse = float(np.mean((pred_np - target_np) ** 2))
    mae = float(np.mean(np.abs(pred_np - target_np)))

    b, time_len, num_leads = pred_np.shape
    corrs = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i in range(b):
            for l in range(num_leads):
                p = pred_np[i, :, l]
                t = target_np[i, :, l]
                p_std = float(np.std(p))
                t_std = float(np.std(t))

                if p_std > 1e-6 and t_std > 1e-6:
                    r = np.corrcoef(p, t)[0, 1]
                    if not np.isnan(r):
                        corrs.append(float(r))
                    else:
                        corrs.append(0.0)
                else:
                    corrs.append(0.0)

    pearson = float(np.mean(corrs)) if len(corrs) > 0 else 0.0

    return {
        "mse": mse,
        "mae": mae,
        "pearson": pearson,
    }


def compute_cnsde_sample_metrics(
    waveform_samples: torch.Tensor,
    ground_truth: torch.Tensor,
    latent_samples: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Computes Section 13 validation and diagnostic metrics over K stochastic samples.

    Args:
        waveform_samples: [B, K, L, C] generated future samples
        ground_truth: [B, L, C] ground truth future
        latent_samples: Optional [B, K, L_z, D] latent trajectories

    Returns:
        Dictionary containing:
            median_pearson, best_of_k_pearson,
            median_mse, best_of_k_mse,
            waveform_sample_std_mean, waveform_sample_std_max,
            latent_sample_std_mean (if latents provided)
    """
    B, K, L, C = waveform_samples.shape
    wf_np = waveform_samples.detach().cpu().numpy()
    gt_np = ground_truth.detach().cpu().numpy()

    median_pearsons = []
    best_pearsons = []
    median_mses = []
    best_mses = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for b in range(B):
            b_pearsons = []
            b_mses = []
            gt_b = gt_np[b]  # [L, C]

            for k in range(K):
                sample_k = wf_np[b, k]  # [L, C]
                mse_k = float(np.mean((sample_k - gt_b) ** 2))
                b_mses.append(mse_k)

                # Pearson across leads and time
                corrs = []
                for c in range(C):
                    p = sample_k[:, c]
                    t = gt_b[:, c]
                    p_std = float(np.std(p))
                    t_std = float(np.std(t))
                    if p_std > 1e-6 and t_std > 1e-6:
                        r = np.corrcoef(p, t)[0, 1]
                        corrs.append(float(r) if not np.isnan(r) else 0.0)
                    else:
                        corrs.append(0.0)
                b_pearsons.append(float(np.mean(corrs)) if corrs else 0.0)

            median_pearsons.append(float(np.median(b_pearsons)))
            best_pearsons.append(float(np.max(b_pearsons)))
            median_mses.append(float(np.median(b_mses)))
            best_mses.append(float(np.min(b_mses)))

    # Diversity metrics: Std_k(x_{b, k, t, c})
    wf_std = waveform_samples.std(dim=1)  # [B, L, C]
    wf_std_mean = float(wf_std.mean().item())
    wf_std_max = float(wf_std.max().item())

    results = {
        "median_pearson": float(np.mean(median_pearsons)) if median_pearsons else 0.0,
        "best_of_k_pearson": float(np.mean(best_pearsons)) if best_pearsons else 0.0,
        "median_mse": float(np.mean(median_mses)) if median_mses else 0.0,
        "best_of_k_mse": float(np.mean(best_mses)) if best_mses else 0.0,
        "waveform_sample_std_mean": wf_std_mean,
        "waveform_sample_std_max": wf_std_max,
    }

    if latent_samples is not None:
        latent_std = latent_samples.std(dim=1)  # [B, L_z, D]
        results["latent_sample_std_mean"] = float(latent_std.mean().item())

    return results
