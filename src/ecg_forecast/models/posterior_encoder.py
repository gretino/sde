from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .context_encoder import Conv1dCausalBlock


class PosteriorEncoder(nn.Module):
    """Bidirectional 1D Residual CNN encoder over context + future waveform.
    Input: [B, 700, num_leads] (500 context + 200 future).
    Downsamples by factor of 4 to 175 tokens @ 25 Hz.
    Extracts 50 recognition path tokens for the future window and posterior initial distribution q(z_0).
    """

    def __init__(self, num_leads: int = 12, context_dim: int = 128, latent_dim: int = 32):
        super().__init__()
        self.num_leads = num_leads
        self.context_dim = context_dim
        self.latent_dim = latent_dim

        # Downsampling architecture
        self.conv1 = nn.Conv1d(num_leads, 64, kernel_size=7, stride=2, padding=3)  # 700 -> 350
        self.res1 = Conv1dCausalBlock(64)
        self.conv2 = nn.Conv1d(64, context_dim, kernel_size=5, stride=2, padding=2)  # 350 -> 175
        self.res2 = Conv1dCausalBlock(context_dim)
        self.res3 = Conv1dCausalBlock(context_dim)

        # Attention pooling over sequence
        self.attn_net = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Posterior projections conditioned on posterior summary + context summary
        self.fc_mean = nn.Linear(2 * context_dim, latent_dim)
        self.fc_logvar = nn.Linear(2 * context_dim, latent_dim)

    def forward(
        self,
        full_waveform: torch.Tensor,
        context_summary: torch.Tensor,
        future_samples: int = 200,
        sampling_rate: int = 100,
        latent_rate: int = 25,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            full_waveform: Concatenated [context, future] waveform [B, T_full, num_leads]
            context_summary: Prior context summary [B, context_dim]
            future_samples: Number of future waveform samples
            sampling_rate: Sampling rate in Hz (default 100)
            latent_rate: Latent sampling rate in Hz (default 25)
        Returns:
            posterior_summary: [B, context_dim]
            recognition_path: [B, future_latent_steps, context_dim]
            posterior_mean: [B, latent_dim]
            posterior_logvar: [B, latent_dim]
        """
        # [B, T_full, num_leads] -> [B, num_leads, T_full]
        x = full_waveform.transpose(1, 2)

        x = F.gelu(self.conv1(x))
        x = self.res1(x)
        x = F.gelu(self.conv2(x))
        x = self.res2(x)
        x = self.res3(x)

        # Transpose back: [B, context_dim, T_tokens] -> [B, T_tokens, context_dim]
        full_tokens = x.transpose(1, 2)

        # Dynamic slicing based on future_samples / (sampling_rate / latent_rate)
        future_seconds = float(future_samples) / float(sampling_rate)
        future_latent_steps = int(round(future_seconds * float(latent_rate)))
        future_latent_steps = min(max(1, future_latent_steps), full_tokens.size(1))

        recognition_path = full_tokens[:, -future_latent_steps:, :]

        # Attention pooling over recognition path
        weights = F.softmax(self.attn_net(recognition_path), dim=1)
        posterior_summary = (recognition_path * weights).sum(dim=1)  # [B, context_dim]

        # Combine posterior summary with context summary for initial state q(z_0)
        combined = torch.cat([posterior_summary, context_summary], dim=-1)
        posterior_mean = self.fc_mean(combined)
        posterior_logvar = self.fc_logvar(combined).clamp(min=-8.0, max=4.0)

        return posterior_summary, recognition_path, posterior_mean, posterior_logvar
