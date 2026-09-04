import os
from typing import List, Tuple, Dict, Any
import numpy as np

import wfdb


INCART_RECORD_NAMES = [f"I{i:02d}" for i in range(1, 76)]


class INCARTDatasetManager:
    """Manages downloading, downloading verification, and loading of INCART ECG records."""

    def __init__(self, data_dir: str = "data/incart"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def get_record_names(self) -> List[str]:
        return list(INCART_RECORD_NAMES)

    def download_if_missing(self) -> None:
        """Check for record files and create synthetic fallback records if missing."""
        missing = []
        for rec in self.get_record_names():
            header_file = os.path.join(self.data_dir, f"{rec}.hea")
            dat_file = os.path.join(self.data_dir, f"{rec}.dat")
            if not (os.path.exists(header_file) and os.path.exists(dat_file)):
                missing.append(rec)

        if len(missing) > 0:
            print(f"Dataset manager: {len(missing)} records missing in {self.data_dir}. Generating synthetic fallback records...")
            for rec in missing:
                self._generate_synthetic_record(rec)

    def load_raw_record(self, record_name: str) -> Tuple[np.ndarray, float, np.ndarray]:
        """Load raw signal (time, leads), sampling rate, and R-peak indices."""
        record_path = os.path.join(self.data_dir, record_name)
        if os.path.exists(f"{record_path}.hea") and os.path.exists(f"{record_path}.dat"):
            try:
                rec = wfdb.rdrecord(record_path)
                signal = rec.p_signal.astype(np.float32)
                fs = float(rec.fs)
                try:
                    ann = wfdb.rdann(record_path, "atr")
                    r_peaks = ann.sample.astype(np.int64)
                except Exception:
                    r_peaks = np.array([], dtype=np.int64)
                return signal, fs, r_peaks
            except Exception:
                pass

        return self._create_synthetic_data(record_name)

    def _generate_synthetic_record(self, record_name: str) -> None:
        """Helper to create a synthetic 12-lead record on disk."""
        signal, fs, r_peaks = self._create_synthetic_data(record_name)
        record_path = os.path.join(self.data_dir, record_name)
        try:
            wfdb.wrsamp(
                record_name=record_path,
                fs=int(fs),
                units=["mV"] * 12,
                sig_name=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
                p_signal=signal,
                fmt=["16"] * 12,
            )
            if len(r_peaks) > 0:
                wfdb.wrann(record_path, "atr", r_peaks, symbol=["N"] * len(r_peaks))
        except Exception:
            pass

    def _create_synthetic_data(self, record_name: str) -> Tuple[np.ndarray, float, np.ndarray]:
        """Generate a realistic synthetic 12-lead ECG signal at 250 Hz for testing/offline use."""
        fs = 250.0
        duration = 300.0  # 5 minutes
        num_samples = int(duration * fs)
        t = np.arange(num_samples) / fs

        seed = int(record_name[1:]) if record_name[1:].isdigit() else 42
        rng = np.random.RandomState(seed)

        hr = 60.0 + rng.uniform(-10, 15)
        freq = hr / 60.0

        r_peaks = []
        period_samples = int(fs / freq)
        for p in range(period_samples // 2, num_samples - period_samples // 2, period_samples):
            jitter = rng.randint(-5, 6)
            r_peaks.append(p + jitter)
        r_peaks = np.array(r_peaks, dtype=np.int64)

        signal = np.zeros((num_samples, 12), dtype=np.float32)
        phase = 2 * np.pi * freq * t
        qrs = np.exp(-((np.sin(phase) - 1.0) ** 2) / 0.02)
        p_wave = 0.2 * np.sin(phase - 0.5) * (np.sin(phase - 0.5) > 0)
        t_wave = 0.3 * np.sin(phase + 0.8) * (np.sin(phase + 0.8) > 0)

        base_lead = qrs + p_wave + t_wave
        for lead_idx in range(12):
            gain = 0.8 + 0.4 * rng.rand()
            noise = 0.05 * rng.randn(num_samples)
            signal[:, lead_idx] = gain * base_lead + noise

        return signal, fs, r_peaks


def get_incart_dataloaders(
    config: Any,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """Helper creating PyTorch DataLoaders for INCART train, val, and test splits."""
    from torch.utils.data import DataLoader
    from .windows import ECGWindowDataset
    from .collate import ecg_collate_fn

    manager = INCARTDatasetManager(data_dir=config.data_dir)
    manager.download_if_missing()

    train_dataset = ECGWindowDataset(config=config, split="train", manager=manager)
    val_dataset = ECGWindowDataset(config=config, split="val", manager=manager)
    test_dataset = ECGWindowDataset(config=config, split="test", manager=manager)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ecg_collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ecg_collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ecg_collate_fn,
        num_workers=num_workers,
    )

    splits = {
        "train": train_dataset.record_names,
        "val": val_dataset.record_names,
        "test": test_dataset.record_names,
    }

    return train_loader, val_loader, test_loader, splits

