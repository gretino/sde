import os
import random
import torch
from torch.utils.data import Dataset
import wfdb
from sde.preprocessing import preprocess_ecg
from sde.dataset import SegmentBuilder

class IncartDataset(Dataset):
    """
    PyTorch Dataset for the PhysioNet INCART 12-lead ECG dataset.
    Loads recordings, cleans/resamples them to target_sr (default 100 Hz),
    and builds context and target segments.
    """
    def __init__(
        self,
        db_dir: str,
        record_names: list,
        context_window: float = 10.0,
        prediction_window: float = 10.0,
        target_sr: int = 100,
        cache_dir: str = "cache",
        use_cache: bool = True
    ):
        self.db_dir = db_dir
        self.record_names = record_names
        self.context_window = context_window
        self.prediction_window = prediction_window
        self.target_sr = target_sr
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        self.signals = {}
        self.annotations = {}
        
        # Load and preprocess all records
        for rec in self.record_names:
            self._load_record(rec)
            
        self.segment_builder = SegmentBuilder(
            sampling_rate=target_sr,
            context_window=context_window,
            prediction_window=prediction_window
        )
        
        self.context_pts = self.segment_builder.context_pts
        self.target_pts = self.segment_builder.target_pts
        self.segment_len_pts = self.context_pts + self.target_pts
        
        # Build list of all non-overlapping segments
        self.samples = []
        for rec in self.record_names:
            sig_len = self.signals[rec].shape[0]
            num_segs = sig_len // self.segment_len_pts
            for i in range(num_segs):
                start_idx = i * self.segment_len_pts
                self.samples.append((rec, start_idx))

    def _load_record(self, rec: str) -> None:
        cache_path = os.path.join(self.cache_dir, f"{rec}_clean.pt")
        ann_cache_path = os.path.join(self.cache_dir, f"{rec}_ann.pt")
        
        if self.use_cache and os.path.exists(cache_path) and os.path.exists(ann_cache_path):
            self.signals[rec] = torch.load(cache_path)
            self.annotations[rec] = torch.load(ann_cache_path)
            return
            
        rec_path = os.path.join(self.db_dir, rec)
        record = wfdb.rdrecord(rec_path)
        
        raw_signal = torch.tensor(record.p_signal, dtype=torch.float32)
        original_sr = record.fs
        
        # Clean and resample using preprocessing module
        clean_signal, _ = preprocess_ecg(raw_signal, original_sr, self.target_sr)
        
        # Load annotations and rescale R-peak samples to target_sr
        ann = wfdb.rdann(rec_path, "atr")
        ann_samples_original = torch.tensor(ann.sample, dtype=torch.float32)
        ann_samples_target = (ann_samples_original * (self.target_sr / original_sr)).round().long()
        
        if self.use_cache:
            torch.save(clean_signal, cache_path)
            torch.save(ann_samples_target, ann_cache_path)
            
        self.signals[rec] = clean_signal
        self.annotations[rec] = ann_samples_target

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        rec, start_idx = self.samples[idx]
        waveform = self.signals[rec]
        
        context_wf = waveform[start_idx : start_idx + self.context_pts]
        target_wf = waveform[start_idx + self.context_pts : start_idx + self.segment_len_pts]
        
        # Local z-score normalization per lead to guarantee mean=0 and std=1 per segment
        context_mean = context_wf.mean(dim=0, keepdim=True)
        context_std = context_wf.std(dim=0, keepdim=True)
        context_std = torch.where(context_std > 0, context_std, torch.ones_like(context_std))
        context_wf = (context_wf - context_mean) / context_std
        
        target_mean = target_wf.mean(dim=0, keepdim=True)
        target_std = target_wf.std(dim=0, keepdim=True)
        target_std = torch.where(target_std > 0, target_std, torch.ones_like(target_std))
        target_wf = (target_wf - target_mean) / target_std
        
        dt = 1.0 / self.target_sr
        # Context ends at t=0
        context_start_t = -(self.context_pts - 1) * dt
        context_t = torch.linspace(context_start_t, 0.0, self.context_pts)
        
        # Target begins immediately after t=0
        target_end_t = self.target_pts * dt
        target_t = torch.linspace(dt, target_end_t, self.target_pts)
        
        # Extract R-peaks within this segment's target window
        ann_samples = self.annotations[rec]
        target_start = start_idx + self.context_pts
        target_end = start_idx + self.segment_len_pts
        
        mask = (ann_samples >= target_start) & (ann_samples < target_end)
        segment_r_peaks = ann_samples[mask] - target_start # relative to target window start
        
        return context_wf, context_t, target_wf, target_t, segment_r_peaks

def get_incart_splits(db_dir: str, seed: int = 42) -> tuple[list[str], list[str]]:
    """
    Reads the RECORDS list, shuffles them reproducibly with seed, and splits them
    into 50 train and 25 test records.
    """
    records_file = os.path.join(db_dir, "RECORDS")
    with open(records_file, "r") as f:
        records = [line.strip() for line in f if line.strip()]
        
    # Shuffle reproducibly
    rng = random.Random(seed)
    rng.shuffle(records)
    
    train_records = records[:50]
    test_records = records[50:75] # ensure exactly 25
    
    return train_records, test_records
