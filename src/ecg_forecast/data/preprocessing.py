import os
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from scipy.signal import resample_poly
from math import gcd

PREPROCESSING_VERSION = "v1.0.0"


def preprocess_record(
    signal: np.ndarray,
    orig_fs: float,
    r_peaks: np.ndarray,
    target_fs: float = 100.0,
    lead_indices: Optional[List[int]] = None,
    record_id: str = "record",
    cache_dir: Optional[str] = "cache/preprocessed",
) -> Tuple[np.ndarray, np.ndarray]:
    """Resamples signal to target_fs, adjusts R-peak indices, handles lead selection, and caches to disk."""

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        lead_tag = "all" if lead_indices is None else "_".join(map(str, lead_indices))
        cache_filename = f"{record_id}_fs{int(target_fs)}_{lead_tag}_{PREPROCESSING_VERSION}.npz"
        cache_path = os.path.join(cache_dir, cache_filename)

        if os.path.exists(cache_path):
            data = np.load(cache_path)
            return data["signal"].astype(np.float32), data["r_peaks"].astype(np.int64)

    # Resample signal using rational polyphase resampling
    orig_fs_int = int(round(orig_fs))
    target_fs_int = int(round(target_fs))

    if orig_fs_int != target_fs_int:
        g = gcd(orig_fs_int, target_fs_int)
        up = target_fs_int // g
        down = orig_fs_int // g
        resampled_signal = resample_poly(signal, up=up, down=down, axis=0).astype(np.float32)

        # Scale R-peak indices
        ratio = target_fs / orig_fs
        resampled_r_peaks = np.round(r_peaks * ratio).astype(np.int64)
        num_resampled = resampled_signal.shape[0]
        resampled_r_peaks = resampled_r_peaks[(resampled_r_peaks >= 0) & (resampled_r_peaks < num_resampled)]
    else:
        resampled_signal = signal.astype(np.float32)
        resampled_r_peaks = r_peaks.astype(np.int64)

    # Lead selection if specified
    if lead_indices is not None:
        resampled_signal = resampled_signal[:, lead_indices]

    if cache_dir is not None:
        np.savez_compressed(cache_path, signal=resampled_signal, r_peaks=resampled_r_peaks)

    return resampled_signal, resampled_r_peaks
