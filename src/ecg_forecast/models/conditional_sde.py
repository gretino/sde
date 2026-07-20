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

        # Learnable diagonal diffusion initialized to ~0.03 (softplus(-3.5) + 1e-4 approx 0.0303)
        self.raw_sigma = nn.Parameter(torch.full((latent_dim,), -3.5))

        # Context and recognition features passed before sdeint call
        self._mode = "posterior"
        self._context_summary: Optional[torch.Tensor] = None
        self._recognition_path: Optional[torch.Tensor] = None

    def set_context(
        self,
        mode: str,
        context_summary: torch.Tensor,
        recognition_path: Optional[torch.Tensor] = None,
    ):
        self._mode = mode
        self._context_summary = context_summary
        self._recognition_path = recognition_path

    @property
    def sigma(self) -> torch.Tensor:
        return 1e-4 + F.softplus(self.raw_sigma)

    def _interpolate_recognition(self, t: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        if self._recognition_path is None:
            return torch.zeros((batch_size, self.context_dim), device=device)

        # Map t in [0, 2.0] to float index in [0, 49]
        t_scalar = float(t) if isinstance(t, (int, float)) else t.item() if t.numel() == 1 else float(t[0])
        idx_float = (t_scalar / 2.0) * 49.0
        idx_low = int(math.floor(idx_float))
        idx_high = min(idx_low + 1, 49)
        weight_high = idx_float - idx_low
        weight_low = 1.0 - weight_high

        rec_low = self._recognition_path[:, idx_low, :]
        rec_high = self._recognition_path[:, idx_high, :]
        return weight_low * rec_low + weight_high * rec_high

    def h(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Prior drift h(t, z)."""
        b = z.size(0)
        t_norm = (t.expand(b, 1) if t.dim() > 0 else t.repeat(b).unsqueeze(-1)) / 2.0
        c = self._context_summary if self._context_summary is not None else torch.zeros((b, self.context_dim), device=z.device)
        inp = torch.cat([z, t_norm, c], dim=-1)
        return self.prior_drift_net(inp)

    def f(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Posterior drift f(t, z)."""
        b = z.size(0)
        t_norm = (t.expand(b, 1) if t.dim() > 0 else t.repeat(b).unsqueeze(-1)) / 2.0
        c = self._context_summary if self._context_summary is not None else torch.zeros((b, self.context_dim), device=z.device)
        rec_t = self._interpolate_recognition(t, b, z.device)
        inp = torch.cat([z, t_norm, c, rec_t], dim=-1)
        return self.posterior_drift_net(inp)

    def g(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Diagonal diffusion g(t, z)."""
        b = z.size(0)
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

    def integrate(
        self,
        z0: torch.Tensor,
        ts: torch.Tensor,
        context_summary: torch.Tensor,
        recognition_path: Optional[torch.Tensor] = None,
        mode: str = "posterior",
        brownian_motion: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z0: Initial latent states [B, latent_dim]
            ts: Timestamps [50] (0.04, 0.08, ..., 2.00)
            context_summary: Context summary [B, context_dim]
            recognition_path: Optional recognition path [B, 50, context_dim]
            mode: 'posterior' or 'prior'
        Returns:
            latent_path: Trajectory [B, 50, latent_dim]
            path_kl: Scalar Girsanov Path KL divergence term
        """
        self.sde_func.set_context(mode=mode, context_summary=context_summary, recognition_path=recognition_path)

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
            dt_step = 2.0 / num_steps
            diff_sq_sum = 0.0

            for k in range(num_steps):
                tk = ts[k]
                zk = latent_path[:, k, :]
                f_k = self.sde_func.f(tk, zk)
                h_k = self.sde_func.h(tk, zk)
                drift_diff = (f_k - h_k) / sigma.unsqueeze(0)
                diff_sq_sum = diff_sq_sum + 0.5 * torch.sum(drift_diff ** 2, dim=-1)

            path_kl = (diff_sq_sum * dt_step).mean()

        return latent_path, path_kl
