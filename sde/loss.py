import torch
import torch.nn as nn

class LatentDynamicsLoss(nn.Module):
    """
    Primary training objective (L_latent_dynamics) that enforces that the continuous
    latent space accurately models physiological progression over time.
    Compares the solver's predicted future latent state against the target encoder's representation
    of the actual future target segment.
    """
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.mse = nn.MSELoss()

    def forward(self, predicted_latent: torch.Tensor, target_waveform: torch.Tensor, target_times: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicted_latent: [batch, latent_dim] predicted latent state at target end-time
            target_waveform: [batch, target_time, leads] raw future target waveform
            target_times: [target_time] future target timestamps
        Returns:
            loss: Scalar tensor representing L_latent_dynamics
        """
        # Encode actual future target segment
        target_latent = self.encoder(target_waveform, target_times) # [batch, latent_dim]
        return self.mse(predicted_latent, target_latent)

class PhaseTolerantWaveformLoss(nn.Module):
    """
    Auxiliary reconstruction objective mapping sparsely queried states z_t into local
    waveform frames. Prioritizes morphology, event timing, and inter-lead relationships
    over absolute point-wise phase alignment.
    """
    def __init__(
        self,
        w_smooth_l1: float = 1.0,
        w_spectral: float = 1.0,
        w_event: float = 1.0,
        w_interlead: float = 1.0
    ):
        super().__init__()
        self.w_smooth_l1 = w_smooth_l1
        self.w_spectral = w_spectral
        self.w_event = w_event
        self.w_interlead = w_interlead
        
        self.smooth_l1 = nn.SmoothL1Loss()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, predicted_waveform: torch.Tensor, target_waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicted_waveform: [batch, time, leads]
            target_waveform: [batch, time, leads]
        Returns:
            loss: Scalar tensor representing the composite loss
        """
        # 1. Smooth L1 Loss (Overall Shape / Alignment)
        loss_smooth_l1 = self.smooth_l1(predicted_waveform, target_waveform)
        
        # 2. Spectral Morphology Loss (L_spectral_morphology)
        pred_fft = torch.fft.rfft(predicted_waveform, dim=1)
        target_fft = torch.fft.rfft(target_waveform, dim=1)
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        loss_spectral = self.l1(pred_mag, target_mag)
        
        # 3. Event Timing Loss (L_event_timing via squared signal envelope)
        pred_env = predicted_waveform ** 2
        target_env = target_waveform ** 2
        loss_event = self.l1(pred_env, target_env)
        
        # 4. Inter-lead Consistency Loss (L_interlead_consistency via cross-lead covariance)
        # Center the signals along the time dimension
        pred_centered = predicted_waveform - predicted_waveform.mean(dim=1, keepdim=True)
        target_centered = target_waveform - target_waveform.mean(dim=1, keepdim=True)
        time_dim = predicted_waveform.shape[1]
        
        # cov = (X_centered.T @ X_centered) / (N - 1)
        # Batch matrix multiplication over leads dimension
        pred_cov = torch.bmm(pred_centered.transpose(1, 2), pred_centered) / (time_dim - 1)
        target_cov = torch.bmm(target_centered.transpose(1, 2), target_centered) / (time_dim - 1)
        loss_interlead = self.mse(pred_cov, target_cov)
        
        # Composite Loss
        total_loss = (
            self.w_smooth_l1 * loss_smooth_l1 +
            self.w_spectral * loss_spectral +
            self.w_event * loss_event +
            self.w_interlead * loss_interlead
        )
        return total_loss
