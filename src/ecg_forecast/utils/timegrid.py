import torch


def make_latent_times(
    future_samples: int,
    sampling_rate: int = 100,
    latent_rate: int = 25,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Centralized timestamp creation for latent SDE trajectory integration.

    Args:
        future_samples: Number of waveform samples in future target (e.g. 50, 100, 200)
        sampling_rate: Waveform sampling rate in Hz (default 100)
        latent_rate: Latent sampling rate in Hz (default 25)
        device: Torch device
        dtype: Torch dtype

    Returns:
        ts: Timestamps tensor of size [future_latent_steps] starting at (1 / latent_rate)
    """
    future_seconds = float(future_samples) / float(sampling_rate)
    future_latent_steps = int(round(future_seconds * latent_rate))
    dt_latent = 1.0 / float(latent_rate)
    start_t = dt_latent
    end_t = future_seconds
    return torch.linspace(start_t, end_t, future_latent_steps, device=device, dtype=dtype)
