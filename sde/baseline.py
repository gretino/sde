import torch
import torch.nn as nn
from sde.solver import ContinuousSolver
from sde.encoder import PhysiologicalEncoder

import numpy as np

class SirenLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, is_first: bool = False, w0: float = 30.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.w0 = w0
        self.is_first = is_first
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # Initialize standard latent channels
                nn.init.uniform_(self.linear.weight, -1 / self.linear.in_features, 1 / self.linear.in_features)
                # Overwrite the last channel (time coordinate) with a coordinate-appropriate scale
                nn.init.uniform_(self.linear.weight[:, -1], -1.0, 1.0)
            else:
                limit = np.sqrt(6 / self.linear.in_features) / self.w0
                nn.init.uniform_(self.linear.weight, -limit, limit)
            if self.linear.bias is not None:
                nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))

class PhaseTolerantDecoder(nn.Module):
    def __init__(self, latent_dim: int, leads: int, hidden_dim: int = 256, w0: float = 30.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.leads = leads
        
        in_features = latent_dim + 1 # Concatenate time coordinate
        
        self.layer1 = SirenLayer(in_features, hidden_dim, is_first=True, w0=w0)
        self.layer2 = SirenLayer(hidden_dim, hidden_dim, is_first=False, w0=1.0)
        self.layer3 = SirenLayer(hidden_dim, hidden_dim, is_first=False, w0=1.0)
        self.out_layer = nn.Linear(hidden_dim, leads)
        
        with torch.no_grad():
            limit = np.sqrt(6 / hidden_dim)
            nn.init.uniform_(self.out_layer.weight, -limit, limit)
            if self.out_layer.bias is not None:
                nn.init.constant_(self.out_layer.bias, 0.0)

    def forward(self, latent_trajectory: torch.Tensor, query_times: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_trajectory: [batch, num_queries, latent_dim]
            query_times: [num_queries] or [batch, num_queries] or [batch, num_queries, 1]
        Returns:
            predicted_waveform: [batch, num_queries, leads]
        """
        batch_size, num_queries, _ = latent_trajectory.shape
        
        # Format query_times to [batch, num_queries, 1]
        if query_times.dim() == 1:
            t = query_times.unsqueeze(0).unsqueeze(-1).expand(batch_size, num_queries, 1)
        elif query_times.dim() == 2:
            t = query_times.unsqueeze(-1)
        else:
            t = query_times
            
        # Scale time to [-1, 1] range to be well-behaved coordinate inputs for SIREN
        t_min, t_max = t.min(), t.max()
        t_scaled = 2.0 * (t - t_min) / (t_max - t_min + 1e-8) - 1.0
        
        x = torch.cat([latent_trajectory, t_scaled], dim=-1)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out = self.out_layer(x)
        return out

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
