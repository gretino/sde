import pytest
import numpy as np
from scripts.overfit_deterministic_prior import select_distributed_tiny_windows


class DummyDataset:
    def __len__(self):
        return 1000


def test_tiny_window_sampler():
    dataset = DummyDataset()
    indices = select_distributed_tiny_windows(dataset, target_count=32)

    assert len(indices) == 32
    assert indices[0] == 0
    assert indices[-1] == 999
    # Ensure indices are distributed across the dataset span, not range(32)
    assert indices != list(range(32)), "Tiny window sampler returned contiguous indices from record 0!"
