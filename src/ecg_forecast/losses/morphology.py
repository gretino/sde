from typing import Tuple, Dict, Any
import torch


def compute_derivative_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss between temporal first-differences along waveform time dimension.
    pred: [B, 200, num_leads]
    target: [B, 200, num_leads]
    """
    diff_pred = pred[:, 1:, :] - pred[:, :-1, :]
    diff_target = target[:, 1:, :] - target[:, :-1, :]
    return torch.abs(diff_pred - diff_target).mean()


def compute_spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Multi-resolution STFT magnitude L1 loss (FFT 32, 64, 128; Hop 8, 16, 32)."""
    fft_sizes = [32, 64, 128]
    hop_sizes = [8, 16, 32]

    b, time_len, leads = pred.shape
    total_spectral_loss = torch.tensor(0.0, device=pred.device)

    # Transpose to [B * leads, time_len]
    pred_flat = pred.transpose(1, 2).reshape(b * leads, time_len)
    target_flat = target.transpose(1, 2).reshape(b * leads, time_len)

    for n_fft, hop in zip(fft_sizes, hop_sizes):
        window = torch.hann_window(n_fft, device=pred.device)
        stft_pred = torch.stft(
            pred_flat,
            n_fft=n_fft,
            hop_length=hop,
            window=window,
            return_complex=True,
        )
        stft_target = torch.stft(
            target_flat,
            n_fft=n_fft,
            hop_length=hop,
            window=window,
            return_complex=True,
        )

        mag_pred = torch.abs(stft_pred)
        mag_target = torch.abs(stft_target)

        loss_res = torch.abs(mag_pred - mag_target).mean()
        total_spectral_loss = total_spectral_loss + loss_res

    return total_spectral_loss / len(fft_sizes)


def compute_morphology_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_derivative: float = 0.5,
    lambda_spectral: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combines derivative loss and multi-resolution STFT spectral loss."""
    deriv_loss = compute_derivative_loss(pred, target)
    spectral_loss = compute_spectral_loss(pred, target)

    total_morphology = lambda_derivative * deriv_loss + lambda_spectral * spectral_loss

    metrics = {
        "derivative_loss": float(deriv_loss.item()),
        "spectral_loss": float(spectral_loss.item()),
        "morphology_loss": float(total_morphology.item()),
    }

    return total_morphology, metrics
