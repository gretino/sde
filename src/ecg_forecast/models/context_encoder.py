from typing import Tuple, Dict
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
    """Boundary-Aware Causal 1D Residual CNN context encoder.
    Downsamples 500-sample context window @ 100 Hz to 125 tokens @ 25 Hz.
    Outputs global_summary, boundary_token, and recent_summary for cardiac boundary phase preservation.
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

        # Dynamic context projection from [global, boundary, recent] (3 * context_dim) to context_dim
        self.dynamic_proj = nn.Linear(context_dim * 3, context_dim)

        # Projections to prior initial state p(z_0)
        self.fc_mean = nn.Linear(context_dim, latent_dim)
        self.fc_logvar = nn.Linear(context_dim, latent_dim)

    def encode_features(self, context_waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extracts component context features for phase probing and dynamic conditioning.
        Args:
            context_waveform: [B, T_c, num_leads]
        Returns:
            dict containing 'tokens', 'global', 'boundary', 'recent', 'dynamic'
        """
        # [B, T_c, num_leads] -> [B, num_leads, T_c]
        x = context_waveform.transpose(1, 2)

        x = F.gelu(self.conv1(x))
        x = self.res1(x)
        x = F.gelu(self.conv2(x))
        x = self.res2(x)
        x = self.res3(x)

        # Transpose back: [B, context_dim, T_tokens] -> [B, T_tokens, context_dim]
        context_tokens = x.transpose(1, 2)

        # Attention pooling
        weights = F.softmax(self.attn_net(context_tokens), dim=1)  # [B, T_tokens, 1]
        global_summary = (context_tokens * weights).sum(dim=1)     # [B, context_dim]

        # Boundary token (at t=0) and recent summary (final 1.0s / 25 tokens)
        boundary_token = context_tokens[:, -1, :]
        num_recent = min(25, context_tokens.size(1))
        recent_summary = context_tokens[:, -num_recent:, :].mean(dim=1)

        concat_feats = torch.cat([global_summary, boundary_token, recent_summary], dim=-1)
        dynamic_summary = F.gelu(self.dynamic_proj(concat_feats))

        return {
            "tokens": context_tokens,
            "global": global_summary,
            "boundary": boundary_token,
            "recent": recent_summary,
            "dynamic": dynamic_summary,
        }

    def get_dynamic_context(self, context_tokens: torch.Tensor, global_summary: torch.Tensor) -> torch.Tensor:
        """Constructs boundary-aware dynamic context c_dynamic = [global_summary, boundary_token, recent_summary]."""
        boundary_token = context_tokens[:, -1, :]  # Final context token at t=0
        num_recent = min(25, context_tokens.size(1))
        recent_summary = context_tokens[:, -num_recent:, :].mean(dim=1)  # Mean over final 1 second
        concat_feats = torch.cat([global_summary, boundary_token, recent_summary], dim=-1)
        return F.gelu(self.dynamic_proj(concat_feats))

    def forward(self, context_waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            context_waveform: [B, 500, num_leads]
        Returns:
            context_summary: [B, context_dim] (dynamic boundary-aware summary)
            context_tokens: [B, 125, context_dim]
            prior_mean: [B, latent_dim]
            prior_logvar: [B, latent_dim]
        """
        feats = self.encode_features(context_waveform)
        context_summary = feats["dynamic"]
        context_tokens = feats["tokens"]

        # Prior initial state parameterization
        prior_mean = self.fc_mean(context_summary)
        prior_logvar = self.fc_logvar(context_summary).clamp(min=-8.0, max=4.0)

        return context_summary, context_tokens, prior_mean, prior_logvar
