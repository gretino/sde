import torch
import pytest
import numpy as np

def test_preprocess_ecg_cleans_raw_waveform():
    # We test the public interface behavior, not implementation
    # A dummy noisy "ECG-like" tensor [batch, time, leads] or just [time, leads]
    # Let's say [time, leads]
    original_sr = 100
    time_points = 500
    
    # Create a noisy signal
    t = np.linspace(0, 5, time_points)
    clean_signal = np.sin(2 * np.pi * 1.0 * t)  # 1 Hz sine wave
    noise = np.random.normal(0, 0.5, time_points)
    noisy_signal = clean_signal + noise
    
    # Convert to torch tensor: shape [500, 1]
    waveform = torch.tensor(noisy_signal, dtype=torch.float32).unsqueeze(1)
    
    import sde.preprocessing
    
    # Tracer bullet: just clean without resampling
    clean_tensor, timestamps = sde.preprocessing.preprocess_ecg(
        waveform=waveform, 
        original_sr=original_sr, 
        target_sr=original_sr
    )
    
    # Assertions
    assert isinstance(clean_tensor, torch.Tensor)
    assert isinstance(timestamps, torch.Tensor)
    assert clean_tensor.shape == waveform.shape
    assert timestamps.shape == (time_points,)
    # The cleaned signal should be standardized (mean ~0, var ~1)
    assert torch.allclose(torch.mean(clean_tensor), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(torch.var(clean_tensor, unbiased=False), torch.tensor(1.0), atol=1e-5)
    # Timestamps should be properly spaced
    assert torch.allclose(timestamps[1] - timestamps[0], torch.tensor(1.0 / original_sr))
