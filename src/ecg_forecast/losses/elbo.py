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
    scale: [num_leads] or [num_gpus, num_leads] or [num_gpus * num_leads] (when gathered by DataParallel)
    """
    num_leads = pred_mean.size(-1)
    if scale.numel() > num_leads:
        scale_vec = scale.view(-1, num_leads).mean(dim=0)
    elif scale.dim() > 1:
        scale_vec = scale.mean(dim=0)
    else:
        scale_vec = scale

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


def compute_initial_teacher_loss(
    prior_mean: torch.Tensor,
    prior_logvar: torch.Tensor,
    post_mean_detached: torch.Tensor,
    post_logvar_detached: torch.Tensor,
) -> torch.Tensor:
    """Computes KL(stopgrad(q(z0)) || p(z0)) averaged over batch and latent dimensions (Section 8.7)."""
    var_p = torch.exp(prior_logvar)
    var_q = torch.exp(post_logvar_detached)
    kl = 0.5 * (
        prior_logvar - post_logvar_detached + (var_q + (post_mean_detached - prior_mean) ** 2) / var_p - 1.0
    )
    return kl.mean(dim=-1).mean()


def compute_drift_teacher_loss(
    prior_drift: torch.Tensor,
    post_drift_detached: torch.Tensor,
) -> torch.Tensor:
    """Computes L2 loss ||h_theta(t, stopgrad(z_t^q)) - stopgrad[f_phi(t, z_t^q)]||^2 averaged over batch, time, and latent dim (Section 8.8)."""
    diff_sq = (prior_drift - post_drift_detached) ** 2
    return diff_sq.mean()


def compute_autonomous_trajectory_loss(
    prior_latent_path: torch.Tensor,
    post_latent_path_detached: torch.Tensor,
) -> torch.Tensor:
    """Computes autonomous trajectory matching loss L_trajectory = (1/BTD) ||z_{1:T}^p - stopgrad(z_{1:T}^q)||_2^2."""
    diff_sq = (prior_latent_path - post_latent_path_detached.detach()) ** 2
    return diff_sq.mean()


def compute_initial_mean_loss(
    prior_mean: torch.Tensor,
    post_mean_detached: torch.Tensor,
) -> torch.Tensor:
    """Computes direct initial-state mean matching loss L_{z_0} = (1/BD) ||\mu_p - stopgrad(\mu_q)||_2^2."""
    diff_sq = (prior_mean - post_mean_detached.detach()) ** 2
    return diff_sq.mean()


def compute_weighted_kl_ratio(
    weighted_initial_kl: float,
    weighted_path_kl: float,
    waveform_objective: float,
) -> float:
    """Computes weighted_kl_ratio = (weighted_initial_kl + weighted_path_kl) / max(1e-8, waveform_objective) (Section 4.3)."""
    denom = max(1e-8, abs(waveform_objective))
    return float((weighted_initial_kl + weighted_path_kl) / denom)

