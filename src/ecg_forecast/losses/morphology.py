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
    """Multi-resolution STFT magnitude L1 loss with adaptive FFT sizes for short horizons."""
    b, time_len, leads = pred.shape
    candidate_pairs = [(16, 4), (32, 8), (64, 16), (128, 32)]
    valid_pairs = [(f, h) for f, h in candidate_pairs if f <= time_len]
    if len(valid_pairs) == 0:
        n_f = max(8, time_len // 2)
        valid_pairs = [(n_f, max(1, n_f // 4))]

    total_spectral_loss = torch.tensor(0.0, device=pred.device)

    # Transpose to [B * leads, time_len]
    pred_flat = pred.transpose(1, 2).reshape(b * leads, time_len)
    target_flat = target.transpose(1, 2).reshape(b * leads, time_len)

    for n_fft, hop in valid_pairs:
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

    return total_spectral_loss / len(valid_pairs)



def create_soft_rpeak_target(
    r_peaks_list: list,
    time_len: int = 200,
    sigma_samples: float = 3.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Creates soft Gaussian R-peak target map [B, time_len] for annotated R-peak locations."""
    b = len(r_peaks_list)
    targets = torch.zeros(b, time_len, device=device)
    t_idx = torch.arange(time_len, device=device).float().unsqueeze(0)  # [1, time_len]

    for i, peaks in enumerate(r_peaks_list):
        if len(peaks) > 0:
            for p in peaks:
                p_val = float(p)
                if 0 <= p_val < time_len:
                    g = torch.exp(-0.5 * ((t_idx - p_val) / sigma_samples) ** 2).squeeze(0)
                    targets[i] = torch.maximum(targets[i], g)

    return targets


def compute_rhythm_loss(
    pred_r_logits: torch.Tensor,
    r_peaks_list: list,
    sigma_samples: float = 3.0,
) -> torch.Tensor:
    """Computes AMP-safe Binary Cross Entropy with logits loss between predicted R-peak logits and soft Gaussian targets."""
    b, time_len = pred_r_logits.shape
    target_map = create_soft_rpeak_target(r_peaks_list, time_len=time_len, sigma_samples=sigma_samples, device=pred_r_logits.device)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred_r_logits.float(), target_map.float())
    return bce



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

