from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmissionDecoder(nn.Module):
    """Decodes 25 Hz latent states into 100 Hz 12-lead ECG waveforms (4 sub-samples per latent step)."""

    def __init__(self, num_leads: int = 12, latent_dim: int = 32, context_dim: int = 128):
        super().__init__()
        self.num_leads = num_leads
        self.latent_dim = latent_dim
        self.context_dim = context_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim + context_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 4 * num_leads),
        )

        # Learned observation log-scale per lead initialized to softplus parameter ~0.1
        self.raw_obs_log_scale = nn.Parameter(torch.full((num_leads,), -2.3))

    @property
    def observation_scale(self) -> torch.Tensor:
        return (1e-3 + F.softplus(self.raw_obs_log_scale)).clamp(min=0.01, max=2.0)

    def forward(self, latent_path: torch.Tensor, context_summary: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent_path: [B, 50, 32]
            context_summary: [B, 128]
        Returns:
            waveform_mean: [B, 200, num_leads]
            waveform_scale: [num_leads]
        """
        b, steps, d = latent_path.shape  # B, 50, 32

        # Expand context summary to match temporal steps: [B, 50, 128]
        c_expanded = context_summary.unsqueeze(1).expand(-1, steps, -1)

        # Combine: [B, 50, 32 + 128]
        inp = torch.cat([latent_path, c_expanded], dim=-1)

        # Decode each step: [B, 50, 4 * num_leads]
        out_sub = self.net(inp)

        # Reshape to [B, 50, 4, num_leads] -> flatten temporal dimension to [B, 200, num_leads]
        out_reshaped = out_sub.view(b, steps, 4, self.num_leads)
        waveform_mean = out_reshaped.view(b, steps * 4, self.num_leads)

        return waveform_mean, self.observation_scale
