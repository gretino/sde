from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch

from ..metrics.waveform import compute_waveform_metrics
from ..metrics.rhythm import compute_rhythm_metrics, detect_r_peaks


def compute_waveform_debug_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    lead_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Computes comprehensive waveform fidelity metrics for 12-lead ECG.
    
    pred, target: [B, T, C] or [T, C]
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    b, t, c = pred.shape
    if lead_names is None:
        lead_names = [f"Lead_{i+1}" for i in range(c)]

    diff = pred - target
    mse = float(torch.mean(diff ** 2).item())
    mae = float(torch.mean(torch.abs(diff)).item())

    # Per-lead Pearson and errors
    lead_pearsons = []
    lead_mses = []
    lead_maes = []
    lead_ranges = []
    lead_stds = []

    for l_idx in range(c):
        p_l = pred[:, :, l_idx].detach().cpu().numpy().reshape(-1)
        t_l = target[:, :, l_idx].detach().cpu().numpy().reshape(-1)

        l_mse = float(np.mean((p_l - t_l) ** 2))
        l_mae = float(np.mean(np.abs(p_l - t_l)))
        lead_mses.append(l_mse)
        lead_maes.append(l_mae)

        p_range = float(np.ptp(p_l))
        p_std = float(np.std(p_l))
        lead_ranges.append(p_range)
        lead_stds.append(p_std)

        if np.std(p_l) > 1e-6 and np.std(t_l) > 1e-6:
            r = float(np.corrcoef(p_l, t_l)[0, 1])
            if np.isnan(r):
                r = 0.0
        else:
            r = 0.0
        lead_pearsons.append(r)

    macro_pearson = float(np.mean(lead_pearsons))
    median_pearson = float(np.median(lead_pearsons))

    # Derivative MAE
    pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
    target_diff = target[:, 1:, :] - target[:, :-1, :]
    deriv_mae = float(torch.mean(torch.abs(pred_diff - target_diff)).item())

    # Waveform temporal std and amplitude range overall
    waveform_temporal_std = float(pred.std(dim=1).mean().item())
    waveform_amplitude_range = float((pred.max(dim=1)[0] - pred.min(dim=1)[0]).mean().item())

    per_lead_dict = {}
    for i, name in enumerate(lead_names):
        per_lead_dict[name] = {
            "mse": lead_mses[i],
            "mae": lead_maes[i],
            "pearson": lead_pearsons[i],
            "amplitude_range": lead_ranges[i],
            "temporal_std": lead_stds[i],
        }

    return {
        "mse": mse,
        "mae": mae,
        "macro_pearson": macro_pearson,
        "median_pearson": median_pearson,
        "derivative_mae": deriv_mae,
        "waveform_temporal_std": waveform_temporal_std,
        "waveform_amplitude_range": waveform_amplitude_range,
        "per_lead": per_lead_dict,
    }


def compute_rhythm_debug_metrics(
    pred_waveform: torch.Tensor,
    target_waveform: torch.Tensor,
    sampling_rate: int = 100,
    r_peak_tolerance_ms: float = 50.0,
) -> Dict[str, float]:
    """Computes R-peak detection, timing, and heart-rate metrics for 12-lead forecast.
    
    pred_waveform, target_waveform: [B, T, C]
    """
    b, t, c = pred_waveform.shape
    # Use Lead II (idx 1 if 12 lead, else idx 0) for rhythm analysis
    lead_idx = 1 if c >= 2 else 0

    pred_np = pred_waveform[:, :, lead_idx].detach().cpu().numpy()
    target_np = target_waveform[:, :, lead_idx].detach().cpu().numpy()

    precision_list, recall_list, f1_list = [], [], []
    hr_error_list = []
    next_peak_timing_error_list = []
    zero_peak_count = 0

    tol_samples = int(round((r_peak_tolerance_ms / 1000.0) * sampling_rate))

    for i in range(b):
        p_peaks = detect_r_peaks(pred_np[i], fs=sampling_rate)
        t_peaks = detect_r_peaks(target_np[i], fs=sampling_rate)

        if len(p_peaks) == 0:
            zero_peak_count += 1

        # R-peak match precision/recall/f1
        matched_t = set()
        tp = 0
        for p in p_peaks:
            for t_idx, t_p in enumerate(t_peaks):
                if t_idx not in matched_t and abs(p - t_p) <= tol_samples:
                    matched_t.add(t_idx)
                    tp += 1
                    break

        prec = tp / len(p_peaks) if len(p_peaks) > 0 else 0.0
        rec = tp / len(t_peaks) if len(t_peaks) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)

        # Heart rate MAE
        p_hr = (len(p_peaks) / (t / sampling_rate)) * 60.0
        t_hr = (len(t_peaks) / (t / sampling_rate)) * 60.0
        hr_error_list.append(abs(p_hr - t_hr))

        # Next R-peak timing MAE (in ms)
        if len(p_peaks) > 0 and len(t_peaks) > 0:
            first_p_ms = (p_peaks[0] / sampling_rate) * 1000.0
            first_t_ms = (t_peaks[0] / sampling_rate) * 1000.0
            next_peak_timing_error_list.append(abs(first_p_ms - first_t_ms))

    return {
        "rpeak_precision": float(np.mean(precision_list)),
        "rpeak_recall": float(np.mean(recall_list)),
        "rpeak_f1": float(np.mean(f1_list)),
        "zero_rpeak_forecast_pct": float((zero_peak_count / b) * 100.0),
        "heart_rate_mae": float(np.mean(hr_error_list)),
        "next_rpeak_timing_mae_ms": float(np.mean(next_peak_timing_error_list)) if len(next_peak_timing_error_list) > 0 else 999.0,
    }


def compute_latent_debug_metrics(
    prior_latent_path: torch.Tensor,
    teacher_latent_path: Optional[torch.Tensor] = None,
    prior_mean: Optional[torch.Tensor] = None,
    post_mean: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Computes latent trajectory dynamics metrics.
    
    prior_latent_path: [B, T_latent, D]
    """
    b, t, d = prior_latent_path.shape

    # Path length L_path = (1/B) sum_b sum_t ||z_{t+1} - z_t||_2
    step_diffs = prior_latent_path[:, 1:, :] - prior_latent_path[:, :-1, :]
    path_length = float(torch.norm(step_diffs, dim=-1).sum(dim=1).mean().item())

    latent_temporal_std = float(prior_latent_path.std(dim=1).mean().item())
    latent_deriv_norm = float(torch.norm(step_diffs, dim=-1).mean().item())

    res = {
        "latent_path_length": path_length,
        "latent_temporal_std": latent_temporal_std,
        "latent_derivative_norm": latent_deriv_norm,
    }

    if prior_mean is not None and post_mean is not None:
        init_dist = float(torch.norm(prior_mean - post_mean, dim=-1).mean().item())
        res["initial_state_mean_distance"] = init_dist

    if teacher_latent_path is not None:
        traj_mse = float(torch.mean((prior_latent_path - teacher_latent_path) ** 2).item())
        res["trajectory_teacher_mse"] = traj_mse

    return res


def compute_uncertainty_debug_metrics(
    samples_waveform: torch.Tensor,
    target_waveform: torch.Tensor,
    samples_latent: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Computes multisample uncertainty metrics across N forecast samples for a single or batch of contexts.
    
    samples_waveform: [N, B, T, C] or [N, T, C]
    target_waveform: [B, T, C] or [T, C]
    """
    if samples_waveform.dim() == 3:
        samples_waveform = samples_waveform.unsqueeze(1)  # [N, 1, T, C]
    if target_waveform.dim() == 2:
        target_waveform = target_waveform.unsqueeze(0)  # [1, T, C]

    n, b, t, c = samples_waveform.shape

    # Empirical waveform variance across samples [B, T, C]
    sample_var = samples_waveform.var(dim=0)
    mean_sample_std = float(torch.sqrt(sample_var).mean().item())

    # Pointwise 90% interval width and coverage (5th - 95th percentile)
    q05 = torch.quantile(samples_waveform, 0.05, dim=0)
    q95 = torch.quantile(samples_waveform, 0.95, dim=0)

    interval_width = float((q95 - q05).mean().item())
    max_interval_width = float((q95 - q05).max().item())

    inside_mask = (target_waveform >= q05) & (target_waveform <= q95)
    interval_coverage = float(inside_mask.float().mean().item())

    # Pairwise sample distance
    pairwise_dists = []
    for i in range(min(n, 16)):
        for j in range(i + 1, min(n, 16)):
            d = torch.mean(torch.abs(samples_waveform[i] - samples_waveform[j]))
            pairwise_dists.append(float(d.item()))
    mean_pairwise_dist = float(np.mean(pairwise_dists)) if len(pairwise_dists) > 0 else 0.0

    res = {
        "mean_waveform_sample_std": mean_sample_std,
        "interval_width_90": interval_width,
        "max_interval_width_90": max_interval_width,
        "interval_coverage_90": interval_coverage,
        "mean_pairwise_sample_distance": mean_pairwise_dist,
    }

    if samples_latent is not None:
        # samples_latent: [N, B, T_latent, D]
        if samples_latent.dim() == 3:
            samples_latent = samples_latent.unsqueeze(1)
        z0_var = samples_latent[:, :, 0, :].var(dim=0).mean().item()
        zT_var = samples_latent[:, :, -1, :].var(dim=0).mean().item()
        res["latent_z0_sample_var"] = float(z0_var)
        res["latent_zT_sample_var"] = float(zT_var)
        res["latent_variance_retention"] = float(zT_var / (z0_var + 1e-8))

    return res
