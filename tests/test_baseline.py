import torch
import pytest

from sde.baseline import NeuroSDEBaseline, PhysiologicalEncoder, PhaseTolerantDecoder
from sde.solver import ContinuousSolver, FBase

def test_neuro_sde_forward_pass_shape():
    """
    Tracer Bullet: Proves the end-to-end pipeline connects without crashing.
    Given a batch of context waveforms and future query times, it should output
    a predicted waveform and a latent trajectory of the correct shapes.
    """
    batch_size = 2
    context_len = 100
    leads = 12
    latent_dim = 32
    num_queries = 5
    conv_layers = [(256, 2, 2)] * 4

    # Mock components
    encoder = PhysiologicalEncoder(in_leads=leads, conv_layers=conv_layers, latent_dim=latent_dim)
    f_base = FBase(latent_dim=latent_dim, hidden_dim=32)
    solver = ContinuousSolver(f_base=f_base)
    decoder = PhaseTolerantDecoder(latent_dim=latent_dim, leads=leads)

    model = NeuroSDEBaseline(encoder, solver, decoder)

    # Dummy inputs
    context_waveform = torch.randn(batch_size, context_len, leads)
    context_times = torch.linspace(-1.0, 0.0, context_len) # Normalized Anchor Time: ends at 0
    query_times = torch.linspace(0.1, 1.0, num_queries) # Future offsets > 0

    predicted_waveform, latent_trajectory = model(context_waveform, context_times, query_times)

    # Verify shapes
    # predicted_waveform: [batch, num_queries, leads]
    assert predicted_waveform.shape == (batch_size, num_queries, leads)
    # latent_trajectory: [batch, num_queries, latent_dim]
    assert latent_trajectory.shape == (batch_size, num_queries, latent_dim)

    # Ensure gradients flow back to the input context waveform and parameters
    context_waveform.requires_grad = True
    predicted_waveform, latent_trajectory = model(context_waveform, context_times, query_times)
    loss = predicted_waveform.sum() + latent_trajectory.sum()
    loss.backward()

    assert context_waveform.grad is not None
    # Check that model parameters (encoder, solver, decoder) have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} did not receive gradients"
