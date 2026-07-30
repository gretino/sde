import math
from typing import Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsde


class SDEFunc(nn.Module):
    """Itô SDE definition compatible with torchsde.sdeint.
    Can operate in 'posterior' mode (uses f drift) or 'prior' mode (uses h drift).
    """

    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, latent_dim: int = 32, context_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim

        # Prior drift h(t, z, c)
        self.prior_drift_net = nn.Sequential(
            nn.Linear(latent_dim + 1 + context_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, latent_dim),
        )

        # Posterior drift f(t, z, c, rec_t)
        self.posterior_drift_net = nn.Sequential(
            nn.Linear(latent_dim + 1 + context_dim + context_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, latent_dim),
        )

        # Learnable diagonal diffusion for Stage C
        self.raw_sigma = nn.Parameter(torch.full((latent_dim,), 0.0))
        self.register_buffer("fixed_sigma", torch.full((latent_dim,), 0.01))
        self.stage = "A"

        # Context and recognition features passed before sdeint call
        self._mode = "posterior"
        self._deterministic: bool = False
        self._context_summary: Optional[torch.Tensor] = None
        self._recognition_path: Optional[torch.Tensor] = None
        self._recognition_duration: float = 2.0

    def set_stage(self, stage: str):
        self.stage = stage.upper()

    def set_context(
        self,
        mode: str,
        context_summary: torch.Tensor,
        recognition_path: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        recognition_duration: float = 2.0,
    ):
        self._mode = mode
        self._context_summary = context_summary
        self._recognition_path = recognition_path
        self._deterministic = deterministic
        self._recognition_duration = recognition_duration

    @property
    def sigma(self) -> torch.Tensor:
        if self.stage in ["A", "B"]:
            # Fixed diffusion 0.01 during Stage A and B (Section 5)
            return self.fixed_sigma
        else:
            # Stage C bounded learnable diffusion: 0.005 + 0.045 * sigmoid(raw_sigma) -> range [0.005, 0.050]
            return 0.005 + 0.045 * torch.sigmoid(self.raw_sigma)

    def _interpolate_recognition(self, t: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        if self._recognition_path is None:
            return torch.zeros((batch_size, self.context_dim), device=device)

        num_steps = self._recognition_path.size(1)
        if num_steps <= 1:
            return self._recognition_path[:, 0, :]

        duration = max(self._recognition_duration, 1e-5)
        t_scalar = float(t) if isinstance(t, (int, float)) else (t.item() if t.numel() == 1 else float(t[0]))
        idx_float = (t_scalar / duration) * float(num_steps - 1)
        idx_low = min(max(0, int(math.floor(idx_float))), num_steps - 1)
        idx_high = min(num_steps - 1, idx_low + 1)
        weight_high = idx_float - float(idx_low)
        weight_low = 1.0 - weight_high

        rec_low = self._recognition_path[:, idx_low, :]
        rec_high = self._recognition_path[:, idx_high, :]
        return weight_low * rec_low + weight_high * rec_high

    def h(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Prior drift h(t, z)."""
        b = z.size(0)
        t_norm = (t.expand(b, 1) if t.dim() > 0 else t.repeat(b).unsqueeze(-1)) / max(self._recognition_duration, 1e-5)
        c = self._context_summary if self._context_summary is not None else torch.zeros((b, self.context_dim), device=z.device)
        inp = torch.cat([z, t_norm, c], dim=-1)
        return self.prior_drift_net(inp)

    def f(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Posterior drift f(t, z)."""
        b = z.size(0)
        t_norm = (t.expand(b, 1) if t.dim() > 0 else t.repeat(b).unsqueeze(-1)) / max(self._recognition_duration, 1e-5)
        c = self._context_summary if self._context_summary is not None else torch.zeros((b, self.context_dim), device=z.device)
        rec_t = self._interpolate_recognition(t, b, z.device)
        inp = torch.cat([z, t_norm, c, rec_t], dim=-1)
        return self.posterior_drift_net(inp)

    def g(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Diagonal diffusion g(t, z). Returns exact 0 when deterministic=True."""
        b = z.size(0)
        if self._deterministic:
            return torch.zeros((b, self.latent_dim), device=z.device, dtype=z.dtype)
        return self.sigma.unsqueeze(0).expand(b, -1)

    def f_eval(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self._mode == "prior":
            return self.h(t, z)
        else:
            return self.f(t, z)


class ConditionalLatentSDE(nn.Module):
    """Integrates conditional latent SDE trajectories via torchsde.sdeint."""

    def __init__(self, latent_dim: int = 32, context_dim: int = 128, dt: float = 0.01):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.dt = dt
        self.sde_func = SDEFunc(latent_dim=latent_dim, context_dim=context_dim)

    def set_stage(self, stage: str):
        self.sde_func.set_stage(stage)

    def integrate(
        self,
        z0: torch.Tensor,
        ts: torch.Tensor,
        context_summary: torch.Tensor,
        recognition_path: Optional[torch.Tensor] = None,
        mode: str = "posterior",
        brownian_motion: Optional[Any] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z0: Initial latent states [B, latent_dim]
            ts: Timestamps [future_latent_steps]
            context_summary: Context summary [B, context_dim]
            recognition_path: Optional recognition path [B, future_latent_steps, context_dim]
            mode: 'posterior' or 'prior'
            brownian_motion: Optional Brownian motion object
            deterministic: If True, sets diffusion g(t,z)=0 for exact deterministic integration
        Returns:
            latent_path: Trajectory [B, future_latent_steps, latent_dim]
            path_kl: Scalar Girsanov Path KL divergence term
        """
        duration = float(ts[-1].item()) if ts.numel() > 0 else 2.0
        self.sde_func.set_context(
            mode=mode,
            context_summary=context_summary,
            recognition_path=recognition_path,
            deterministic=deterministic,
            recognition_duration=duration,
        )

        class SDEWrapper(nn.Module):
            sde_type = "ito"
            noise_type = "diagonal"

            def __init__(self, func, m):
                super().__init__()
                self.func = func
                self.m = m

            def f(self, t, z):
                return self.func.f_eval(t, z)

            def g(self, t, z):
                return self.func.g(t, z)

        wrapper = SDEWrapper(self.sde_func, mode)

        if ts[0] > 0:
            ts_int = torch.cat([torch.tensor([0.0], device=ts.device, dtype=ts.dtype), ts])
            prepend = True
        else:
            ts_int = ts
            prepend = False

        trajectory = torchsde.sdeint(
            sde=wrapper,
            y0=z0,
            ts=ts_int,
            method="euler",
            dt=self.dt,
            bm=brownian_motion,
        )

        if prepend:
            trajectory = trajectory[1:]

        latent_path = trajectory.transpose(0, 1)

        path_kl = torch.tensor(0.0, device=z0.device)
        if mode == "posterior" and recognition_path is not None:
            sigma = self.sde_func.sigma
            num_steps = ts.size(0)
            diff_sq_sum = 0.0

            for k in range(num_steps):
                tk = ts[k]
                zk = latent_path[:, k, :]
                f_k = self.sde_func.f(tk, zk)
                h_k = self.sde_func.h(tk, zk)
                drift_ratio_sq = ((f_k - h_k) / sigma.unsqueeze(0)).pow(2)
                diff_sq_sum = diff_sq_sum + drift_ratio_sq.mean()

            # Normalized path KL over batch, time, and latent dimensions
            path_kl = 0.5 * (diff_sq_sum / float(num_steps)) * duration

        return latent_path, path_kl


    def _interpolate_recognition(self, recognition_path: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:

        self.sde_func._recognition_path = recognition_path
        batch_size = recognition_path.size(0)
        device = recognition_path.device
        num_steps = ts.size(0)
        interp_tokens = []
        for k in range(num_steps):
            tk = ts[k]
            token_k = self.sde_func._interpolate_recognition(tk, batch_size, device)
            interp_tokens.append(token_k)
        return torch.stack(interp_tokens, dim=1)


# Alias for backward compatibility and testing
ConditionalSDE = ConditionalLatentSDE


