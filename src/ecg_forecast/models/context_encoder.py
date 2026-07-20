from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dCausalBlock(nn.Module):
    """Residual 1D Conv block with GELU activations."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return self.act(out + res)


class ContextEncoder(nn.Module):
    """Causal 1D Residual CNN context encoder.
    Downsamples 500-sample context window @ 100 Hz to 125 tokens @ 25 Hz.
    Feature channel width is controlled by context_dim (e.g. 64 or 128).
    """

    def __init__(self, num_leads: int = 12, context_dim: int = 128, latent_dim: int = 32):
        super().__init__()
        self.num_leads = num_leads
        self.context_dim = context_dim
        self.latent_dim = latent_dim

        # Downsampling architecture
        self.conv1 = nn.Conv1d(num_leads, 64, kernel_size=7, stride=2, padding=3)  # 500 -> 250
        self.res1 = Conv1dCausalBlock(64)
        self.conv2 = nn.Conv1d(64, context_dim, kernel_size=5, stride=2, padding=2)  # 250 -> 125
        self.res2 = Conv1dCausalBlock(context_dim)
        self.res3 = Conv1dCausalBlock(context_dim)

        # Attention pooling over 125 context tokens
        self.attn_net = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Projections to prior initial state p(z_0)
        self.fc_mean = nn.Linear(context_dim, latent_dim)
        self.fc_logvar = nn.Linear(context_dim, latent_dim)

    def forward(self, context_waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            context_waveform: [B, 500, num_leads]
        Returns:
            context_summary: [B, context_dim]
            context_tokens: [B, 125, context_dim]
            prior_mean: [B, latent_dim]
            prior_logvar: [B, latent_dim]
        """
        # [B, 500, num_leads] -> [B, num_leads, 500]
        x = context_waveform.transpose(1, 2)

        x = F.gelu(self.conv1(x))
        x = self.res1(x)
        x = F.gelu(self.conv2(x))
        x = self.res2(x)
        x = self.res3(x)

        # Transpose back: [B, context_dim, 125] -> [B, 125, context_dim]
        context_tokens = x.transpose(1, 2)

        # Attention pooling
        weights = F.softmax(self.attn_net(context_tokens), dim=1)  # [B, 125, 1]
        context_summary = (context_tokens * weights).sum(dim=1)    # [B, context_dim]

        # Prior initial state parameterization
        prior_mean = self.fc_mean(context_summary)
        prior_logvar = self.fc_logvar(context_summary).clamp(min=-8.0, max=4.0)

        return context_summary, context_tokens, prior_mean, prior_logvar
