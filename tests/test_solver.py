import torch
import pytest

from sde.solver import ContinuousSolver, FBase

def test_solver_extrapolates_latent_trajectory():
    """
    Tests that ContinuousSolver correctly uses torchdiffeq and FBase
    to integrate from t=0 across the target query times.
    """
    batch_size = 4
    latent_dim = 16
    num_queries = 10
    
    # query_times must include t=0 internally or be strictly positive?
    # Usually odeint takes a 1D tensor of times [t0, t1, t2...]
    # Our target times don't include t=0. The solver must start at t=0
    # and integrate to the query times. We can test this behavior.
    query_times = torch.linspace(0.1, 1.0, num_queries)
    
    f_base = FBase(latent_dim=latent_dim, hidden_dim=32)
    solver = ContinuousSolver(f_base=f_base)
    
    # Initial physiological state (at t=0)
    z_t0 = torch.randn(batch_size, latent_dim)
    
    # We expect the output trajectory to match the query_times exactly
    # excluding t=0 if it's not requested.
    latent_trajectory = solver(z_t0, query_times)
    
    # Output shape should be [batch, num_queries, latent_dim]
    assert latent_trajectory.shape == (batch_size, num_queries, latent_dim)
    
    # Ensure gradients can flow through the solver back to z_t0
    z_t0.requires_grad = True
    latent_trajectory = solver(z_t0, query_times)
    loss = latent_trajectory.sum()
    loss.backward()
    
    assert z_t0.grad is not None
