import torch
import pytest
from sde.encoder import PhysiologicalEncoder
from sde.loss import LatentDynamicsLoss, PhaseTolerantWaveformLoss

def test_latent_dynamics_loss():
    """
    Validates that LatentDynamicsLoss computes MSE between predicted latent state
    and encoded target waveform, and that gradients flow back.
    """
    batch_size = 2
    context_len = 100
    leads = 12
    latent_dim = 16
    conv_layers = [(128, 2, 2)] * 2
    
    encoder = PhysiologicalEncoder(in_leads=leads, conv_layers=conv_layers, latent_dim=latent_dim)
    loss_fn = LatentDynamicsLoss(encoder)
    
    predicted_latent = torch.randn(batch_size, latent_dim, requires_grad=True)
    target_waveform = torch.randn(batch_size, context_len, leads)
    target_times = torch.linspace(0.0, 1.0, context_len)
    
    loss = loss_fn(predicted_latent, target_waveform, target_times)
    
    assert loss.dim() == 0  # scalar
    assert loss.item() >= 0
    
    loss.backward()
    assert predicted_latent.grad is not None

def test_phase_tolerant_waveform_loss():
    """
    Validates that PhaseTolerantWaveformLoss computes the composite loss
    and propagates gradients through the target components.
    """
    batch_size = 2
    time_pts = 50
    leads = 12
    
    loss_fn = PhaseTolerantWaveformLoss()
    
    pred_wf = torch.randn(batch_size, time_pts, leads, requires_grad=True)
    target_wf = torch.randn(batch_size, time_pts, leads)
    
    loss = loss_fn(pred_wf, target_wf)
    
    assert loss.dim() == 0
    assert loss.item() >= 0
    
    loss.backward()
    assert pred_wf.grad is not None
