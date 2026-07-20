import os
import pytest
import torch
import numpy as np

from ecg_forecast.config import DataConfig
from ecg_forecast.data import ECGWindowDataset, get_dataset_splits


def test_dataset_splits_isolation(tmp_path):
    records = [f"I{i:02d}" for i in range(1, 76)]
    split_file = str(tmp_path / "splits.json")

    splits = get_dataset_splits(records, seed=42, split_file=split_file)

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    # Disjointness
    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0

    # Coverage
    assert len(train_set) == 50
    assert len(val_set) == 10
    assert len(test_set) == 15


def test_dataset_window_shapes():
    cfg = DataConfig(
        dataset_name="incart",
        data_dir="data/incart",
        num_leads=1,
        lead_indices=[1],
        sampling_rate=100,
        context_duration=5.0,
        future_duration=2.0,
        cache_dir="cache/test_preprocessed",
    )
    ds = ECGWindowDataset(config=cfg, split="train")

    assert len(ds) > 0
    sample = ds[0]

    assert sample["context_waveform"].shape == (500, 1)
    assert sample["future_waveform"].shape == (200, 1)
    assert sample["context_times"].shape == (500,)
    assert sample["future_times"].shape == (200,)
    assert sample["normalization_mean"].shape == (1,)
    assert sample["normalization_std"].shape == (1,)


def test_context_only_normalization():
    cfg = DataConfig(
        dataset_name="incart",
        data_dir="data/incart",
        num_leads=1,
        lead_indices=[1],
        sampling_rate=100,
        cache_dir="cache/test_preprocessed",
    )
    ds = ECGWindowDataset(config=cfg, split="train")
    sample = ds[0]

    ctx_wf = sample["context_waveform"]
    mean = ctx_wf.mean(dim=0)
    std = ctx_wf.std(dim=0)

    # Context waveform should have zero mean and unit variance
    assert torch.abs(mean).item() < 1e-4
    assert torch.abs(std - 1.0).item() < 1e-3
