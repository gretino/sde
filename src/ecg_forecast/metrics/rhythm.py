import warnings
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from scipy.signal import find_peaks

try:
    import neurokit2 as nk
    HAS_NEUROKIT = True
except ImportError:
    HAS_NEUROKIT = False


def detect_r_peaks_lead(signal: np.ndarray, sampling_rate: int = 100) -> np.ndarray:
    """Detects R-peak sample indices in a 1D waveform signal safely."""
    if len(signal) < 10:
        return np.array([], dtype=np.int64)

    if HAS_NEUROKIT and len(signal) >= sampling_rate:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            try:
                _, rpeaks = nk.ecg_peaks(signal, sampling_rate=sampling_rate)
                peaks = rpeaks.get("ECG_R_Peaks", np.array([], dtype=np.int64))
                if len(peaks) > 0:
                    return np.array(peaks, dtype=np.int64)
            except Exception:
                pass

    # Robust fallback peak detection
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        std = float(np.std(signal)) if signal.size > 0 else 0.0
        mean_val = float(np.mean(signal)) if signal.size > 0 else 0.0
        height = mean_val + 1.2 * std if std > 1e-5 else None
        distance = max(1, int(0.4 * sampling_rate))  # min 400 ms distance
        peaks, _ = find_peaks(signal, height=height, distance=distance)
        return np.array(peaks, dtype=np.int64)


def match_peaks(pred_peaks: np.ndarray, target_peaks: np.ndarray, tolerance: int = 5) -> Tuple[int, int, int]:
    """Matches predicted peaks to ground truth target peaks within tolerance sample window.
    Returns (tp, fp, fn).
    """
    if len(target_peaks) == 0 and len(pred_peaks) == 0:
        return 0, 0, 0
    if len(target_peaks) == 0:
        return 0, len(pred_peaks), 0
    if len(pred_peaks) == 0:
        return 0, 0, len(target_peaks)

    matched_target = set()
    tp = 0

    for p in pred_peaks:
        diffs = np.abs(target_peaks - p)
        min_idx = int(np.argmin(diffs))
        if diffs[min_idx] <= tolerance and min_idx not in matched_target:
            tp += 1
            matched_target.add(min_idx)

    fp = len(pred_peaks) - tp
    fn = len(target_peaks) - tp

    return tp, fp, fn


def compute_hr_rmssd(peaks: np.ndarray, sampling_rate: float = 100.0) -> Tuple[float, float]:
    """Computes Heart Rate (BPM) and RMSSD (ms) from peak sample indices safely."""
    if len(peaks) < 2:
        return 0.0, 0.0

    rri_ms = (np.diff(peaks) / sampling_rate) * 1000.0  # ms
    if len(rri_ms) == 0:
        return 0.0, 0.0

    mean_rri = float(np.mean(rri_ms)) if len(rri_ms) > 0 else 0.0
    if mean_rri <= 0:
        return 0.0, 0.0

    hr_bpm = 60000.0 / mean_rri

    if len(rri_ms) >= 2:
        diff_rri = np.diff(rri_ms)
        rmssd_ms = float(np.sqrt(np.mean(diff_rri ** 2))) if len(diff_rri) > 0 else 0.0
    else:
        rmssd_ms = 0.0

    return float(hr_bpm), float(rmssd_ms)


def compute_rhythm_metrics(
    pred_waveforms: torch.Tensor,
    target_r_peaks_list: List[torch.Tensor],
    sampling_rate: int = 100,
    lead_idx: int = 0,
) -> Dict[str, float]:
    """Computes R-peak F1, Heart-Rate MAE, and RMSSD MAE over a batch."""
    pred_np = pred_waveforms.detach().cpu().numpy()
    b = pred_np.shape[0]

    total_tp, total_fp, total_fn = 0, 0, 0
    hr_errors = []
    rmssd_errors = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i in range(b):
            sig = pred_np[i, :, lead_idx]
            pred_peaks = detect_r_peaks_lead(sig, sampling_rate=sampling_rate)

            tgt_peaks_tensor = target_r_peaks_list[i]
            tgt_peaks = tgt_peaks_tensor.cpu().numpy() if isinstance(tgt_peaks_tensor, torch.Tensor) else np.array(tgt_peaks_tensor)

            tp, fp, fn = match_peaks(pred_peaks, tgt_peaks, tolerance=5)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            p_hr, p_rmssd = compute_hr_rmssd(pred_peaks, sampling_rate=sampling_rate)
            t_hr, t_rmssd = compute_hr_rmssd(tgt_peaks, sampling_rate=sampling_rate)

            if t_hr > 0 and p_hr > 0:
                hr_errors.append(abs(p_hr - t_hr))
            if t_rmssd > 0 and p_rmssd > 0:
                rmssd_errors.append(abs(p_rmssd - t_rmssd))

    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    f1 = (2 * precision * recall) / max(1e-5, precision + recall)

    hr_mae = float(np.mean(hr_errors)) if len(hr_errors) > 0 else 0.0
    rmssd_mae = float(np.mean(rmssd_errors)) if len(rmssd_errors) > 0 else 0.0

    return {
        "rpeak_f1": float(f1),
        "hr_mae": hr_mae,
        "rmssd_mae": rmssd_mae,
    }
