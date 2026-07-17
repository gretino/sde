import torch
import torch.nn as nn
from sde.solver import ContinuousSolver
from sde.encoder import PhysiologicalEncoder

class PhaseTolerantDecoder(nn.Module):
    def __init__(self, latent_dim: int, leads: int, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.leads = leads
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, leads)
        )

    def forward(self, latent_trajectory: torch.Tensor, query_times: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            latent_trajectory: [batch, num_queries, latent_dim]
            query_times: Optional [num_queries]
        Returns:
            predicted_waveform: [batch, num_queries, leads]
        """
        return self.net(latent_trajectory)

class NeuroSDEBaseline(nn.Module):
    """
    Core baseline module modeling unconditional physiological dynamics.
    """
    def __init__(self, encoder: PhysiologicalEncoder, solver: ContinuousSolver, decoder: PhaseTolerantDecoder):
        super().__init__()
        self.encoder = encoder
        self.solver = solver
        self.decoder = decoder

    def forward(self, context_waveform: torch.Tensor, context_times: torch.Tensor, query_times: torch.Tensor):
        # 1. Encode context into macro Physiological State
        z_t0 = self.encoder(context_waveform, context_times)
        
        # 2. Evolve state continuously
        latent_trajectory = self.solver(z_t0, query_times)
        
        # 3. Decode into ECG waveform
        predicted_waveform = self.decoder(latent_trajectory, query_times)
        
        return predicted_waveform, latent_trajectory
