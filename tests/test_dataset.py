import torch
import pytest

from sde.dataset import SegmentBuilder

def test_segment_builder_normalizes_anchor_time():
    """
    Validates ADR-0002: Normalized Anchor Time.
    The segment builder must map the end of the context window to t=0,
    ensuring that all target times are strictly positive offsets.
    """
    sampling_rate = 100 # Hz
    total_len = 1000 # 10 seconds
    leads = 12
    
    # Context is 3 seconds, target is 2 seconds
    context_sec = 3.0
    target_sec = 2.0
    
    # Create a dummy waveform
    raw_waveform = torch.arange(total_len).unsqueeze(1).expand(-1, leads).float()
    
    builder = SegmentBuilder(
        sampling_rate=sampling_rate,
        context_window=context_sec,
        prediction_window=target_sec
    )
    
    # Start at 4 seconds into the record
    start_time_sec = 4.0
    
    context_wf, context_t, target_wf, target_t = builder.build_segment(
        raw_waveform, start_time_sec=start_time_sec
    )
    
    # Verify lengths
    expected_context_pts = int(context_sec * sampling_rate)
    expected_target_pts = int(target_sec * sampling_rate)
    
    assert context_wf.shape[0] == expected_context_pts
    assert target_wf.shape[0] == expected_target_pts
    
    # Verify Normalized Anchor Time
    # The last point of context should be exactly at t=0
    assert torch.isclose(context_t[-1], torch.tensor(0.0), atol=1e-5)
    
    # The context should start at -context_sec + dt
    # But conventionally, if context_t[-1] == 0, then context_t[0] == -(context_pts - 1) * dt
    dt = 1.0 / sampling_rate
    expected_start = -(expected_context_pts - 1) * dt
    assert torch.isclose(context_t[0], torch.tensor(expected_start), atol=1e-5)
    
    # All target times must be strictly positive (t > 0)
    assert torch.all(target_t > 0)
    
    # Target starts at dt and ends at target_sec
    assert torch.isclose(target_t[0], torch.tensor(dt), atol=1e-5)
    assert torch.isclose(target_t[-1], torch.tensor(expected_target_pts * dt), atol=1e-5)
    
    # Verify data matches slice correctly
    start_idx = int(start_time_sec * sampling_rate)
    assert torch.allclose(context_wf, raw_waveform[start_idx : start_idx + expected_context_pts])
