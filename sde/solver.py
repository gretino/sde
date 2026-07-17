import torch
import torch.nn as nn
from torchdiffeq import odeint

class FBase(nn.Module):
    """
    Deterministic baseline latent dynamics module (f_base).
    Models how the physiological state evolves over time.
    """
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        ODE derivative function: dz/dt = f_base(t, z).
        torchdiffeq expects signature (t, y).
        """
        return self.net(z)

class ContinuousSolver(nn.Module):
    """
    Evolves the latent state forward from the anchor time to requested timestamps.
    """
    def __init__(self, f_base: nn.Module):
        super().__init__()
        self.f_base = f_base
        
    def forward(self, z_t0: torch.Tensor, query_times: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t0: [batch, latent_dim] representing state at t=0
            query_times: [num_queries] containing positive time offsets
            
        Returns:
            latent_trajectory: [batch, num_queries, latent_dim]
        """
        device = z_t0.device
        query_times = query_times.to(device)
        
        # ODE solvers require the integration time array to start with the time
        # corresponding to the initial condition. For us, this is t=0.
        sorted_times, indices = torch.sort(query_times)
        
        # We prepend t=0.0
        t0 = torch.tensor([0.0], dtype=query_times.dtype, device=device)
        t_eval = torch.cat([t0, sorted_times])
        
        # odeint returns shape [len(t_eval), batch, latent_dim]
        trajectory = odeint(self.f_base, z_t0, t_eval, method='dopri5')
        
        # Slice off the initial state at t=0
        trajectory = trajectory[1:]
        
        # Reorder if original query times were not sorted
        inv_indices = torch.argsort(indices)
        trajectory = trajectory[inv_indices]
        
        # Permute to expected output format [batch, num_queries, latent_dim]
        trajectory = trajectory.permute(1, 0, 2)
        
        return trajectory
