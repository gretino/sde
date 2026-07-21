from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from .context_encoder import ContextEncoder
from .posterior_encoder import PosteriorEncoder
from .conditional_sde import ConditionalLatentSDE
from .emission_decoder import EmissionDecoder
from ..config import ModelConfig


@dataclass
class ForecastOutput:
    waveform_mean: torch.Tensor
    waveform_scale: torch.Tensor
    latent_path: torch.Tensor
    initial_kl: Optional[torch.Tensor] = None
    path_kl: Optional[torch.Tensor] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ForecastOutput":
        return cls(
            waveform_mean=d["waveform_mean"],
            waveform_scale=d["waveform_scale"],
            latent_path=d["latent_path"],
            initial_kl=d.get("initial_kl"),
            path_kl=d.get("path_kl"),
        )


class LatentSDEForecaster(nn.Module):
    """Unified Conditional Latent SDE Forecaster for ECG signals."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.context_dim = config.context_dim
        self.num_leads = config.num_leads
        self.dt = config.dt

        self.context_encoder = ContextEncoder(
            num_leads=self.num_leads,
            context_dim=self.context_dim,
            latent_dim=self.latent_dim,
        )
        self.posterior_encoder = PosteriorEncoder(
            num_leads=self.num_leads,
            context_dim=self.context_dim,
            latent_dim=self.latent_dim,
        )
        self.sde = ConditionalLatentSDE(
            latent_dim=self.latent_dim,
            context_dim=self.context_dim,
            dt=self.dt,
        )
        self.decoder = EmissionDecoder(
            num_leads=self.num_leads,
            latent_dim=self.latent_dim,
            context_dim=self.context_dim,
        )

    def _reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def _gaussian_kl(
        self,
        p_mean: torch.Tensor,
        p_logvar: torch.Tensor,
        q_mean: torch.Tensor,
        q_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Computes KL(q || p) between diagonal Gaussians."""
        var_p = torch.exp(p_logvar)
        var_q = torch.exp(q_logvar)
        kl = 0.5 * (
            p_logvar - q_logvar + (var_q + (q_mean - p_mean) ** 2) / var_p - 1.0
        )
        return kl.mean(dim=-1).mean()

    def forward(
        self,
        context_waveform: torch.Tensor,
        future_waveform: Optional[torch.Tensor] = None,
        context_times: Optional[torch.Tensor] = None,
        future_times: Optional[torch.Tensor] = None,
        mode: str = "posterior",
        num_samples: int = 1,
        brownian_motion: Optional[object] = None,
    ) -> Dict[str, torch.Tensor]:
        """Main PyTorch forward entrypoint returning dict for DataParallel gathering."""
        if mode == "posterior" and future_waveform is not None:
            out = self.forward_posterior(
                context_waveform=context_waveform,
                future_waveform=future_waveform,
                context_times=context_times,
                future_times=future_times,
                brownian_motion=brownian_motion,
            )
        else:
            out = self.forward_prior(
                context_waveform=context_waveform,
                context_times=context_times,
                future_times=future_times,
                num_samples=num_samples,
                brownian_motion=brownian_motion,
            )

        dev = out.waveform_mean.device
        init_kl = out.initial_kl if out.initial_kl is not None else torch.tensor(0.0, device=dev)
        path_kl = out.path_kl if out.path_kl is not None else torch.tensor(0.0, device=dev)

        # Make scalar KLs 1D tensors [1] for DataParallel gathering
        if init_kl.dim() == 0:
            init_kl = init_kl.unsqueeze(0)
        if path_kl.dim() == 0:
            path_kl = path_kl.unsqueeze(0)

        return {
            "waveform_mean": out.waveform_mean,
            "waveform_scale": out.waveform_scale,
            "latent_path": out.latent_path,
            "initial_kl": init_kl,
            "path_kl": path_kl,
        }

    def forward_posterior(
        self,
        context_waveform: torch.Tensor,
        future_waveform: torch.Tensor,
        context_times: Optional[torch.Tensor] = None,
        future_times: Optional[torch.Tensor] = None,
        brownian_motion: Optional[object] = None,
    ) -> ForecastOutput:
        """Reconstructs future waveform using posterior encoder and SDE trajectory."""
        b = context_waveform.size(0)
        device = context_waveform.device

        # Prior context encoding
        c_summary, c_tokens, prior_mean, prior_logvar = self.context_encoder(context_waveform)

        # Concatenate context + future for posterior encoding [B, 700, num_leads]
        full_wf = torch.cat([context_waveform, future_waveform], dim=1)
        post_summary, rec_path, post_mean, post_logvar = self.posterior_encoder(full_wf, c_summary)

        # Initial state KL
        initial_kl = self._gaussian_kl(prior_mean, prior_logvar, post_mean, post_logvar)

        # Sample z_0 from posterior
        z0 = self._reparameterize(post_mean, post_logvar)

        # Build timestamps [50] (0.04, 0.08, ..., 2.00)
        if future_times is not None and future_times.dim() == 1:
            ts = future_times[::4]
        elif future_times is not None and future_times.dim() == 2:
            ts = future_times[0, ::4]
        else:
            ts = torch.linspace(0.04, 2.0, 50, device=device)

        # Continuous SDE integration in posterior mode
        latent_path, path_kl = self.sde.integrate(
            z0=z0,
            ts=ts,
            context_summary=c_summary,
            recognition_path=rec_path,
            mode="posterior",
            brownian_motion=brownian_motion,
        )

        # Emission decoding
        wf_mean, wf_scale = self.decoder(latent_path, c_summary)

        return ForecastOutput(
            waveform_mean=wf_mean,
            waveform_scale=wf_scale,
            latent_path=latent_path,
            initial_kl=initial_kl,
            path_kl=path_kl,
        )

    def forward_prior(
        self,
        context_waveform: torch.Tensor,
        context_times: Optional[torch.Tensor] = None,
        future_times: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        brownian_motion: Optional[object] = None,
    ) -> ForecastOutput:
        """Forecasts future waveform using context prior only."""
        b = context_waveform.size(0)
        device = context_waveform.device

        # Prior context encoding
        c_summary, c_tokens, prior_mean, prior_logvar = self.context_encoder(context_waveform)

        if num_samples > 1:
            # Repeat tensors for multi-sample forecasting
            c_summary = c_summary.repeat_interleave(num_samples, dim=0)
            prior_mean = prior_mean.repeat_interleave(num_samples, dim=0)
            prior_logvar = prior_logvar.repeat_interleave(num_samples, dim=0)

        # Sample z_0 from prior
        z0 = self._reparameterize(prior_mean, prior_logvar)

        if future_times is not None and future_times.dim() == 1:
            ts = future_times[::4]
        elif future_times is not None and future_times.dim() == 2:
            ts = future_times[0, ::4]
        else:
            ts = torch.linspace(0.04, 2.0, 50, device=device)

        # Continuous SDE integration in prior mode
        latent_path, path_kl = self.sde.integrate(
            z0=z0,
            ts=ts,
            context_summary=c_summary,
            mode="prior",
            brownian_motion=brownian_motion,
        )

        # Emission decoding
        wf_mean, wf_scale = self.decoder(latent_path, c_summary)

        return ForecastOutput(
            waveform_mean=wf_mean,
            waveform_scale=wf_scale,
            latent_path=latent_path,
            initial_kl=None,
            path_kl=path_kl,
        )
