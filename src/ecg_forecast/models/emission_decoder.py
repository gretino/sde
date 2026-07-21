from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMResConvBlock(nn.Module):
    """Residual 1D Conv block with Feature-wise Linear Modulation (FiLM) static conditioning."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        # scale, bias: [B, C, 1]
        res = x
        out = self.conv1(x)
        out = out * scale + bias  # FiLM modulation
        out = self.act(out)
        out = self.conv2(out)
        out = out * scale + bias  # FiLM modulation
        return self.act(out + res)


class EmissionDecoder(nn.Module):
    """Temporal Conv1d Decoder for 12-lead ECG waveforms.
    Upsamples 25 Hz latent trajectories [B, T_latent, D] to 100 Hz [B, T_waveform, num_leads]
    using temporal Conv1d blocks with FiLM static context conditioning (Section 11).
    """

    def __init__(self, num_leads: int = 12, latent_dim: int = 32, context_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.num_leads = num_leads
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Projection from latent_dim to hidden_dim
        self.proj_in = nn.Linear(latent_dim, hidden_dim)

        # FiLM generator for static morphology conditioning
        self.film_gen = nn.Linear(context_dim, 2 * hidden_dim)

        # Temporal residual Conv1d blocks
        self.res1 = FiLMResConvBlock(hidden_dim)
        self.res2 = FiLMResConvBlock(hidden_dim)
        self.res3 = FiLMResConvBlock(hidden_dim)

        # Final projection to 12 leads
        self.conv_out = nn.Conv1d(hidden_dim, num_leads, kernel_size=5, padding=2)

        self.raw_obs_log_scale = nn.Parameter(torch.full((num_leads,), 0.0))
        self.stage = "A"

    def set_stage(self, stage: str):
        self.stage = stage.upper()

    @property
    def observation_scale(self) -> torch.Tensor:
        if self.stage in ["A", "B"]:
            # Fixed observation scale 0.10 during Stage A and B (Section 6)
            return torch.full((self.num_leads,), 0.10, device=self.raw_obs_log_scale.device)
        else:
            # Stage C bounded learnable scale: 0.03 + 0.27 * sigmoid(raw_scale) -> range [0.03, 0.30]
            return 0.03 + 0.27 * torch.sigmoid(self.raw_obs_log_scale)

    def forward(
        self,
        latent_path: torch.Tensor,
        context_summary: torch.Tensor,
        target_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent_path: [B, T_latent, latent_dim] (e.g. [B, 50, 32])
            context_summary: [B, context_dim]
            target_len: Optional target waveform length (e.g. 50, 100, 200)
        Returns:
            waveform_mean: [B, T_waveform, num_leads] (e.g. [B, 200, 12])
            waveform_scale: [num_leads]
        """
        b, t_lat, d = latent_path.shape

        # 1. Project latent to hidden_dim: [B, T_latent, hidden_dim]
        x_proj = F.gelu(self.proj_in(latent_path))

        # 2. Transpose to [B, hidden_dim, T_latent]
        x_t = x_proj.transpose(1, 2)

        # 3. Temporal interpolation to 100 Hz timeline (exact target_len if provided)
        t_target = target_len if target_len is not None else t_lat * 4
        x_up = F.interpolate(x_t, size=t_target, mode="linear", align_corners=False)


        # 4. Generate FiLM scale and bias parameters from context summary
        film_params = self.film_gen(context_summary)  # [B, 2 * hidden_dim]
        scale, bias = film_params.chunk(2, dim=-1)     # [B, hidden_dim], [B, hidden_dim]
        scale = scale.unsqueeze(-1)                    # [B, hidden_dim, 1]
        bias = bias.unsqueeze(-1)                      # [B, hidden_dim, 1]

        # 5. Pass through temporal residual Conv1d blocks with FiLM modulation
        h = self.res1(x_up, scale, bias)
        h = self.res2(h, scale, bias)
        h = self.res3(h, scale, bias)

        # 6. Project to waveform leads and transpose to [B, T_waveform, num_leads]
        wf_out = self.conv_out(h)                      # [B, num_leads, 200]
        waveform_mean = wf_out.transpose(1, 2)         # [B, 200, num_leads]

        return waveform_mean, self.observation_scale
