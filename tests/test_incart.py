import os
import torch
import pytest
from sde.incart_dataset import get_incart_splits, IncartDataset
from sde.encoder import PhysiologicalEncoder
from sde.weight_utils import load_pretrained_ecg_fm

DB_DIR = "/home/qfbqt/8TB/datasets/physionet.org/files/incartdb/1.0.0"
WEIGHT_PATH = "/home/qfbqt/8TB/hfcache/hub/models--wanglab--ecg-fm/snapshots/584219ea492cdeef2e19ffbdf9c6ecc874ba427e/mimic_iv_ecg_finetuned.pt"

def test_get_incart_splits():
    """
    Verifies that get_incart_splits splits the 75 records into 50 train and 25 test records
    reproducibly using the seed.
    """
    train_1, test_1 = get_incart_splits(DB_DIR, seed=42)
    train_2, test_2 = get_incart_splits(DB_DIR, seed=42)
    
    assert len(train_1) == 50
    assert len(test_1) == 25
    assert train_1 == train_2
    assert test_1 == test_2

    # Different seed should give different shuffle
    train_diff, test_diff = get_incart_splits(DB_DIR, seed=100)
    assert train_diff != train_1

def test_incart_dataset_loading():
    """
    Verifies that IncartDataset correctly loads records, processes them,
    and returns segments of expected shapes.
    """
    # Use a small subset (just 1 record) for the test to be fast
    records = ["I01"]
    dataset = IncartDataset(
        db_dir=DB_DIR,
        record_names=records,
        context_window=10.0,
        prediction_window=10.0,
        target_sr=100,
        use_cache=True
    )
    
    # A 30 min record at 100Hz = 180000 samples.
    # Each segment is 20s = 2000 samples.
    # Total segments = 180000 // 2000 = 90 segments.
    assert len(dataset) == 90
    
    context_wf, context_t, target_wf, target_t, segment_r_peaks = dataset[0]
    
    # 10s at 100Hz = 1000 points
    assert context_wf.shape == (1000, 12)
    assert context_t.shape == (1000,)
    assert target_wf.shape == (1000, 12)
    assert target_t.shape == (1000,)
    assert segment_r_peaks.dim() == 1
    
    # Check that context_t ends at 0 and target_t starts after 0
    assert torch.isclose(context_t[-1], torch.tensor(0.0))
    assert target_t[0] > 0.0

def test_weight_loading_utility():
    """
    Verifies that load_pretrained_ecg_fm correctly loads weights into the patcher.
    """
    latent_dim = 32
    leads = 12
    conv_layers = [(256, 2, 2)] * 4
    
    encoder = PhysiologicalEncoder(in_leads=leads, conv_layers=conv_layers, latent_dim=latent_dim)
    
    # Save a copy of initial random weights
    orig_conv0_weight = encoder.patcher.conv_layers[0][0].weight.clone()
    
    # Load weights
    load_pretrained_ecg_fm(encoder, WEIGHT_PATH)
    
    # The weights should have changed
    new_conv0_weight = encoder.patcher.conv_layers[0][0].weight
    assert not torch.equal(orig_conv0_weight, new_conv0_weight)
