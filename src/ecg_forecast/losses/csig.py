from typing import Optional
import torch
import torch.nn as nn
from ..signatures.signature import compute_signature_features


class ConditionalSignatureLoss(nn.Module):
    """Primary conditional future signature loss (CSig).

    L_CSig = (1 / B) * sum_b || bar_S_b - S*_b ||^2
    where bar_S_b = (1 / K) * sum_k S(hat_Y_{b,k})
    and S*_b is the precomputed expected future signature target.
    """

    def __init__(
        self,
        depth: int = 4,
        dyadic_depth: int = 2,
        lead_lag: bool = True,
        sigy_mean: Optional[torch.Tensor] = None,
        sigy_std: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.depth = depth
        self.dyadic_depth = dyadic_depth
        self.lead_lag = lead_lag
        self.reduction = reduction

        if sigy_mean is not None and sigy_std is not None:
            self.register_buffer("sigy_mean", sigy_mean.clone().detach())
            self.register_buffer("sigy_std", sigy_std.clone().detach())
        else:
            self.sigy_mean = None
            self.sigy_std = None

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        self.register_buffer("sigy_mean", mean.clone().detach())
        self.register_buffer("sigy_std", std.clone().detach())

    def forward(
        self,
        waveform_samples: torch.Tensor,
        target_conditional_sig: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the CSig loss.

        Args:
            waveform_samples: Tensor of shape [B, K, L, C]
            target_conditional_sig: Precomputed target S*_future of shape [B, sig_dim]

        Returns:
            Scalar loss tensor
        """
        B, K, L, C = waveform_samples.shape

        # Flatten [B, K, L, C] -> [B * K, L, C]
        flat_waveforms = waveform_samples.view(B * K, L, C)

        # Compute signature of all generated samples
        flat_sig = compute_signature_features(
            flat_waveforms,
            depth=self.depth,
            dyadic_depth=self.dyadic_depth,
            lead_lag=self.lead_lag,
            mean=self.sigy_mean,
            std=self.sigy_std,
        )

        sig_dim = flat_sig.shape[-1]
        sample_sigs = flat_sig.view(B, K, sig_dim)

        # Expected generated signature: bar_S_b = (1/K) sum_k S(hat_Y_{b,k})
        expected_sig = sample_sigs.mean(dim=1)  # [B, sig_dim]

        target = target_conditional_sig.to(device=expected_sig.device, dtype=expected_sig.dtype)

        diff = expected_sig - target
        if self.reduction == "mean":
            return (diff**2).mean()
        elif self.reduction == "sum_features":
            return (diff**2).sum(dim=-1).mean()
        else:
            return (diff**2).mean()


def compute_csig_loss(
    waveform_samples: torch.Tensor,
    target_conditional_sig: torch.Tensor,
    depth: int = 4,
    dyadic_depth: int = 2,
    lead_lag: bool = True,
    sigy_mean: Optional[torch.Tensor] = None,
    sigy_std: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Functional interface for CSig loss."""
    loss_fn = ConditionalSignatureLoss(
        depth=depth,
        dyadic_depth=dyadic_depth,
        lead_lag=lead_lag,
        sigy_mean=sigy_mean,
        sigy_std=sigy_std,
        reduction=reduction,
    )
    return loss_fn(waveform_samples, target_conditional_sig)
