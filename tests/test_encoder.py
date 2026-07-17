import torch
import pytest

from sde.encoder import PhysiologicalEncoder

def test_encoder_produces_physiological_state():
    """
    Validates that PhysiologicalEncoder integrates the ECG-FM patcher
    and torchcde to produce a latent initial state z_t0.
    """
    batch_size = 4
    context_len = 1000  # 1000 raw samples
    leads = 12
    latent_dim = 32
    
    # ECG-FM typical config
    # 4 layers of (dim=256, kernel=2, stride=2)
    conv_layers = [(256, 2, 2)] * 4
    
    encoder = PhysiologicalEncoder(
        in_leads=leads,
        conv_layers=conv_layers,
        latent_dim=latent_dim
    )
    
    context_wf = torch.randn(batch_size, context_len, leads)
    
    # Time corresponding to each context point
    context_t = torch.linspace(-10.0, 0.0, context_len)
    
    # Encode
    z_t0 = encoder(context_wf, context_t)
    
    assert z_t0.shape == (batch_size, latent_dim)
    
    # Ensure gradients flow back to the inputs
    context_wf.requires_grad = True
    z_t0 = encoder(context_wf, context_t)
    loss = z_t0.sum()
    loss.backward()
    
    assert context_wf.grad is not None
