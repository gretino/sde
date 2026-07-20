from typing import Tuple, Dict, Any
import torch


def compute_laplace_nll(
    pred_mean: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Computes mean Negative Log-Likelihood under Laplace observation distribution.
    pred_mean: [B, 200, num_leads]
    target: [B, 200, num_leads]
    scale: [num_leads] or [num_gpus, num_leads] (when gathered by DataParallel)
    """
    scale_vec = scale.mean(dim=0) if scale.dim() > 1 else scale
    scale_b = scale_vec.view(1, 1, -1)  # Expand to [1, 1, num_leads]
    diff = torch.abs(target - pred_mean)
    nll = torch.log(2.0 * scale_b) + diff / scale_b
    return nll.mean()


def compute_elbo_loss(
    pred_mean: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    initial_kl: torch.Tensor,
    path_kl: torch.Tensor,
    beta_initial: float = 1.0,
    beta_path: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combines Observation Laplace NLL with weighted Initial KL and Path KL.
    Handles both single-GPU scalars and multi-GPU DataParallel gathered 1D tensors.
    """
    nll = compute_laplace_nll(pred_mean, target, scale)

    init_kl_val = initial_kl.mean() if (initial_kl is not None and initial_kl.numel() > 0) else torch.tensor(0.0, device=pred_mean.device)
    path_kl_val = path_kl.mean() if (path_kl is not None and path_kl.numel() > 0) else torch.tensor(0.0, device=pred_mean.device)

    total_elbo = nll + beta_initial * init_kl_val + beta_path * path_kl_val

    metrics = {
        "nll": float(nll.item()),
        "initial_kl": float(init_kl_val.item()),
        "path_kl": float(path_kl_val.item()),
        "elbo_loss": float(total_elbo.item()),
    }

    return total_elbo, metrics
