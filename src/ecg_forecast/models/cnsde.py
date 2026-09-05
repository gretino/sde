from typing import Tuple, Optional, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsde

from ..config import ModelConfig, SDEConfig


class ContextEncoder(nn.Module):
    """Encodes normalized context signature into condition vector c: [B, 64]."""

    def __init__(self, sig_dim: int = 1020, context_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sig_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, context_dim),
        )

    def forward(self, sig_x: torch.Tensor) -> torch.Tensor:
        return self.net(sig_x)


class InitialStateNetwork(nn.Module):
    """Maps [c, epsilon] to initial latent state z0: [B, 64]."""

    def __init__(self, context_dim: int = 64, noise_dim: int = 16, latent_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        in_dim = context_dim + noise_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, c: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([c, epsilon], dim=-1)
        return self.net(inp)


class SDEFunc(nn.Module):
    """Conditional Stratonovich SDE function with diagonal noise.

    Drift and diffusion receive [z_t, c, t_net] where t_net = t / 2.0.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        context_dim: int = 64,
        drift_hidden: Optional[List[int]] = None,
        diffusion_hidden: Optional[List[int]] = None,
        sigma_min: float = 0.005,
        sigma_max: float = 0.20,
    ):
        super().__init__()
        self.sde_type = "stratonovich"
        self.noise_type = "diagonal"
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        drift_hidden = drift_hidden if drift_hidden is not None else [128, 128, 128]
        diffusion_hidden = diffusion_hidden if diffusion_hidden is not None else [128, 128, 128]

        in_dim = latent_dim + context_dim + 1  # [z, c, t]

        # Drift network: 129 -> 128 -> Tanh -> 128 -> Tanh -> 128 -> Tanh -> 64 -> Tanh
        drift_layers = []
        curr_dim = in_dim
        for h in drift_hidden:
            drift_layers.append(nn.Linear(curr_dim, h))
            drift_layers.append(nn.Tanh())
            curr_dim = h
        drift_layers.append(nn.Linear(curr_dim, latent_dim))
        drift_layers.append(nn.Tanh())
        self.drift_net = nn.Sequential(*drift_layers)

        # Diffusion network: 129 -> 128 -> Tanh -> 128 -> Tanh -> 128 -> Tanh -> 64
        diff_layers = []
        curr_dim = in_dim
        for h in diffusion_hidden:
            diff_layers.append(nn.Linear(curr_dim, h))
            diff_layers.append(nn.Tanh())
            curr_dim = h
        diff_layers.append(nn.Linear(curr_dim, latent_dim))
        self.diffusion_net = nn.Sequential(*diff_layers)

        self._c: Optional[torch.Tensor] = None

    def set_context(self, c: torch.Tensor):
        self._c = c

    def _build_input(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self._c is None:
            raise RuntimeError("Context vector c is not set on SDEFunc. Call set_context() first.")
        B = z.shape[0]
        # Normalize time: t_net = t / 2.0
        t_val = (t / 2.0)
        if not torch.is_tensor(t_val):
            t_val = torch.tensor(t_val, device=z.device, dtype=z.dtype)
        t_net = t_val.expand(B, 1) if t_val.numel() == 1 else t_val.view(B, 1)
        return torch.cat([z, self._c, t_net], dim=-1)

    def f(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Drift f(z_t, c, t)."""
        inp = self._build_input(t, z)
        return self.drift_net(inp)

    def g(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Diffusion g(z_t, c, t). Scaled to [sigma_min, sigma_max]."""
        inp = self._build_input(t, z)
        raw = self.diffusion_net(inp)
        return self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(raw)


class ConditionalNeuralSDE(nn.Module):
    """Conditional Neural SDE generator for ECG forecasting."""

    def __init__(
        self,
        sig_dim: int = 1020,
        context_dim: int = 64,
        latent_dim: int = 64,
        initial_noise_dim: int = 16,
        num_leads: int = 1,
        drift_hidden: Optional[List[int]] = None,
        diffusion_hidden: Optional[List[int]] = None,
        sigma_min: float = 0.005,
        sigma_max: float = 0.20,
        sde_method: str = "reversible_heun",
        adjoint_method: str = "adjoint_reversible_heun",
        dt: float = 0.01,
        t_end: float = 2.0,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.latent_dim = latent_dim
        self.initial_noise_dim = initial_noise_dim
        self.num_leads = num_leads
        self.sde_method = sde_method
        self.adjoint_method = adjoint_method
        self.dt = dt
        self.t_end = t_end

        # Submodules
        self.context_encoder = ContextEncoder(
            sig_dim=sig_dim,
            context_dim=context_dim,
            hidden_dim=128,
        )
        self.initial_state_net = InitialStateNetwork(
            context_dim=context_dim,
            noise_dim=initial_noise_dim,
            latent_dim=latent_dim,
            hidden_dim=128,
        )
        self.sde_func = SDEFunc(
            latent_dim=latent_dim,
            context_dim=context_dim,
            drift_hidden=drift_hidden,
            diffusion_hidden=diffusion_hidden,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        self.readout = nn.Linear(latent_dim, num_leads)

        # Integration time points: 0.00, 0.01, ..., 2.00 (201 points)
        num_steps = int(round(t_end / dt)) + 1
        ts = torch.linspace(0.0, t_end, steps=num_steps)
        self.register_buffer("ts", ts)

    @classmethod
    def from_config(cls, model_cfg: ModelConfig, sde_cfg: SDEConfig, sig_dim: int = 1020) -> "ConditionalNeuralSDE":
        return cls(
            sig_dim=sig_dim,
            context_dim=model_cfg.context_dim,
            latent_dim=model_cfg.latent_dim,
            initial_noise_dim=model_cfg.initial_noise_dim,
            num_leads=model_cfg.num_leads,
            drift_hidden=model_cfg.drift_hidden,
            diffusion_hidden=model_cfg.diffusion_hidden,
            sigma_min=model_cfg.sigma_min,
            sigma_max=model_cfg.sigma_max,
            sde_method=sde_cfg.method,
            adjoint_method=sde_cfg.adjoint_method,
            dt=sde_cfg.dt,
        )

    def forward(
        self,
        sig_x: torch.Tensor,
        y0: torch.Tensor,
        num_samples: int = 8,
        epsilon: Optional[torch.Tensor] = None,
        bm: Optional[Any] = None,
        use_adjoint: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates stochastic future ECG forecasts.

        Args:
            sig_x: Normalized context signature [B, sig_dim]
            y0: Final context observation anchor [B, 1, num_leads] or [B, num_leads]
            num_samples: K Monte Carlo samples per context
            epsilon: Optional explicit initial noise [B, K, initial_noise_dim] or [B * K, initial_noise_dim]
            bm: Optional explicit Brownian motion instance
            use_adjoint: Whether to use adjoint backpropagation (defaults to True if training and grad enabled)

        Returns:
            waveform_samples: [B, K, 200, num_leads]
            latent_samples: [B, K, 201, latent_dim]
        """
        B = sig_x.shape[0]
        K = num_samples
        device = sig_x.device
        dtype = sig_x.dtype

        # 1. Encode context: [B, context_dim]
        c = self.context_encoder(sig_x)

        # 2. Replicate context for K samples: [B * K, context_dim]
        c_rep = c.unsqueeze(1).expand(B, K, -1).contiguous().view(B * K, self.context_dim)

        # 3. Sample initial noise epsilon ~ N(0, I) if not provided
        if epsilon is None:
            eps = torch.randn(B * K, self.initial_noise_dim, device=device, dtype=dtype)
        else:
            if epsilon.ndim == 3:
                eps = epsilon.contiguous().view(B * K, self.initial_noise_dim)
            else:
                eps = epsilon.to(device=device, dtype=dtype)

        # 4. Compute initial latent state: [B * K, latent_dim]
        z0 = self.initial_state_net(c_rep, eps)

        # 5. Integrate Neural SDE
        self.sde_func.set_context(c_rep)
        ts = self.ts.to(device=device, dtype=dtype)

        should_use_adjoint = (self.training and torch.is_grad_enabled()) if use_adjoint is None else use_adjoint

        if should_use_adjoint:
            zs = torchsde.sdeint_adjoint(
                self.sde_func,
                z0,
                ts,
                method=self.sde_method,
                adjoint_method=self.adjoint_method,
                dt=self.dt,
                bm=bm,
            )
        else:
            zs = torchsde.sdeint(
                self.sde_func,
                z0,
                ts,
                method=self.sde_method,
                dt=self.dt,
                bm=bm,
            )

        # zs is [201, B * K, latent_dim] -> permute to [B * K, 201, latent_dim]
        zs = zs.permute(1, 0, 2)

        # 6. Readout raw waveform: [B * K, 201, num_leads]
        raw_waveforms = self.readout(zs)

        # 7. Boundary anchoring:
        # y0 format: [B, 1, num_leads]
        if y0.ndim == 2:
            y0 = y0.unsqueeze(1)
        y0_rep = y0.unsqueeze(1).expand(B, K, 1, self.num_leads).contiguous().view(B * K, 1, self.num_leads)

        # y'_t = y_t - y_0^{gen} + y_0^{context}
        waveform = raw_waveforms - raw_waveforms[:, :1, :] + y0_rep

        # Future waveform: drop t=0 integration anchor point -> [B * K, 200, num_leads]
        future_waveform = waveform[:, 1:, :]

        # 8. Unflatten back to [B, K, ...]
        waveform_samples = future_waveform.view(B, K, -1, self.num_leads)
        latent_samples = zs.view(B, K, -1, self.latent_dim)

        return waveform_samples, latent_samples
