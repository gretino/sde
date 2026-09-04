import os
import json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from .incart import INCARTDatasetManager
from .preprocessing import preprocess_record
from ..config import DataConfig


def get_dataset_splits(
    record_names: List[str],
    seed: int = 42,
    split_file: str = "artifacts/splits/incart_seed42.json",
) -> Dict[str, List[str]]:
    """Splits record names deterministically into 50 train, 10 val, 15 test, saving to JSON artifact."""
    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            return json.load(f)

    os.makedirs(os.path.dirname(split_file), exist_ok=True)
    rng = np.random.RandomState(seed)
    shuffled = list(record_names)
    rng.shuffle(shuffled)

    num_records = len(shuffled)
    if num_records == 75:
        train_recs = sorted(shuffled[:50])
        val_recs = sorted(shuffled[50:60])
        test_recs = sorted(shuffled[60:75])
    else:
        # Generic proportional split for subset/debug runs
        n_train = max(1, int(0.66 * num_records))
        n_val = max(1, int(0.14 * num_records))
        train_recs = sorted(shuffled[:n_train])
        val_recs = sorted(shuffled[n_train : n_train + n_val])
        test_recs = sorted(shuffled[n_train + n_val :])

    splits = {
        "train": train_recs,
        "val": val_recs,
        "test": test_recs,
    }

    with open(split_file, "w") as f:
        json.dump(splits, f, indent=2)

    return splits


class ECGWindowDataset(Dataset):
    """Dataset producing sliding window (context + future) pairs from preprocessed ECG records."""

    def __init__(
        self,
        config: DataConfig,
        split: str = "train",
        manager: Optional[INCARTDatasetManager] = None,
    ):
        self.config = config
        self.split = split
        self.manager = manager if manager is not None else INCARTDatasetManager(config.data_dir)

        # Download raw files if needed
        self.manager.download_if_missing()

        # Get records for this split
        all_records = self.manager.get_record_names()
        split_file = os.path.join("artifacts", "splits", f"incart_seed{config.split_seed}.json")
        splits = get_dataset_splits(all_records, seed=config.split_seed, split_file=split_file)
        self.record_names = splits[split]

        # Extract windows across assigned records
        self.samples = []
        self._build_window_index()

    def _build_window_index(self):
        context_samples = self.config.context_samples
        future_samples = self.config.future_samples
        total_window_samples = self.config.total_samples
        stride_samples = int(round(self.config.stride * self.config.sampling_rate))

        for rec_name in self.record_names:
            raw_sig, orig_fs, raw_peaks = self.manager.load_raw_record(rec_name)
            sig, r_peaks = preprocess_record(
                signal=raw_sig,
                orig_fs=orig_fs,
                r_peaks=raw_peaks,
                target_fs=float(self.config.sampling_rate),
                lead_indices=self.config.lead_indices,
                record_id=rec_name,
                cache_dir=self.config.cache_dir,
            )

            num_samples = len(sig)
            if num_samples < total_window_samples:
                continue

            for start_idx in range(0, num_samples - total_window_samples + 1, stride_samples):
                context_end = start_idx + context_samples
                future_end = context_end + future_samples

                self.samples.append(
                    {
                        "record_id": rec_name,
                        "signal": sig,
                        "r_peaks": r_peaks,
                        "start_idx": start_idx,
                        "context_end": context_end,
                        "future_end": future_end,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        sig = item["signal"]
        r_peaks = item["r_peaks"]

        c_start = item["start_idx"]
        c_end = item["context_end"]
        f_end = item["future_end"]

        # Extract raw waveforms
        context_wf = sig[c_start:c_end]  # [500, num_leads]
        future_wf = sig[c_end:f_end]     # [200, num_leads]

        context_tensor = torch.from_numpy(context_wf).float()
        future_tensor = torch.from_numpy(future_wf).float()

        # Context-only normalization
        mean = context_tensor.mean(dim=0)
        std = context_tensor.std(dim=0).clamp_min(1e-5)

        norm_context = (context_tensor - mean) / std
        norm_future = (future_tensor - mean) / std

        # Time vectors relative to normalized anchor time t=0 (context: -5.0s to 0.0s, future: 0.0s to 2.0s)
        fs = float(self.config.sampling_rate)
        context_times = (torch.arange(-self.config.context_samples, 0, dtype=torch.float32)) / fs
        future_times = (torch.arange(0, self.config.future_samples, dtype=torch.float32)) / fs


        # Find R-peaks falling inside future window (relative 0..199)
        in_future_mask = (r_peaks >= c_end) & (r_peaks < f_end)
        future_r_peaks = r_peaks[in_future_mask] - c_end
        future_r_peaks_tensor = torch.from_numpy(future_r_peaks).long()

        return {
            "record_id": item["record_id"],
            "context_waveform": norm_context,
            "future_waveform": norm_future,
            "raw_context_waveform": context_tensor,
            "raw_future_waveform": future_tensor,
            "context_times": context_times,
            "future_times": future_times,
            "normalization_mean": mean,
            "normalization_std": std,
            "future_r_peaks": future_r_peaks_tensor,
        }
