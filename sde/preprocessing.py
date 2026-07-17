import torch
import neurokit2 as nk
import numpy as np
from typing import Tuple

def preprocess_ecg(waveform: torch.Tensor, original_sr: int, target_sr: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cleans, normalizes, and optionally resamples an ECG waveform.
    
    Args:
        waveform: Tensor of shape [time, leads]
        original_sr: Original sampling rate
        target_sr: Target sampling rate
        
    Returns:
        Tuple of (clean_waveform, timestamps)
    """
    signal_np = waveform.detach().cpu().numpy()
    num_leads = signal_np.shape[1] if len(signal_np.shape) > 1 else 1
    
    clean_leads = []
    
    # 1. Cleaning
    for lead_idx in range(num_leads):
        lead_signal = signal_np[:, lead_idx] if len(signal_np.shape) > 1 else signal_np
        # Use neurokit2 to clean
        clean_lead = nk.ecg_clean(lead_signal, sampling_rate=original_sr, method="neurokit")
        clean_leads.append(clean_lead)
        
    clean_signal_np = np.stack(clean_leads, axis=-1)
    
    # 2. Resampling
    if original_sr != target_sr:
        resampled_leads = []
        for lead_idx in range(num_leads):
            resampled_lead = nk.signal_resample(
                clean_signal_np[:, lead_idx], 
                sampling_rate=original_sr, 
                desired_sampling_rate=target_sr
            )
            resampled_leads.append(resampled_lead)
        clean_signal_np = np.stack(resampled_leads, axis=-1)
        final_sr = target_sr
    else:
        final_sr = original_sr
        
    # 3. Create timestamps
    final_length = clean_signal_np.shape[0]
    timestamps = np.arange(final_length) / final_sr
    
    # 4. Normalize amplitude
    for lead_idx in range(num_leads):
        lead_mean = np.mean(clean_signal_np[:, lead_idx])
        lead_std = np.std(clean_signal_np[:, lead_idx])
        if lead_std > 0:
            clean_signal_np[:, lead_idx] = (clean_signal_np[:, lead_idx] - lead_mean) / lead_std
            
    # Convert back to tensor
    clean_tensor = torch.tensor(clean_signal_np, dtype=torch.float32)
    # If the input was 1D, we should return 1D
    if len(waveform.shape) == 1:
        clean_tensor = clean_tensor.squeeze(-1)
        
    timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32)
    
    return clean_tensor, timestamps_tensor
