from typing import List, Dict, Any
import torch


def ecg_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for ECG window batches."""
    record_ids = [b["record_id"] for b in batch]

    context_waveform = torch.stack([b["context_waveform"] for b in batch], dim=0)
    future_waveform = torch.stack([b["future_waveform"] for b in batch], dim=0)

    raw_context_waveform = torch.stack([b["raw_context_waveform"] for b in batch], dim=0)
    raw_future_waveform = torch.stack([b["raw_future_waveform"] for b in batch], dim=0)

    context_times = torch.stack([b["context_times"] for b in batch], dim=0)
    future_times = torch.stack([b["future_times"] for b in batch], dim=0)

    normalization_mean = torch.stack([b["normalization_mean"] for b in batch], dim=0)
    normalization_std = torch.stack([b["normalization_std"] for b in batch], dim=0)

    future_r_peaks = [b["future_r_peaks"] for b in batch]

    return {
        "record_ids": record_ids,
        "context_waveform": context_waveform,
        "future_waveform": future_waveform,
        "raw_context_waveform": raw_context_waveform,
        "raw_future_waveform": raw_future_waveform,
        "context_times": context_times,
        "future_times": future_times,
        "normalization_mean": normalization_mean,
        "normalization_std": normalization_std,
        "future_r_peaks": future_r_peaks,
    }
