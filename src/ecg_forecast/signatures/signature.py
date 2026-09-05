import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_signature_dim(
    input_channels: int = 1,
    depth: int = 4,
    dyadic_depth: int = 2,
    lead_lag: bool = True,
) -> int:
    """Calculates the total signature feature dimension."""
    d = input_channels + 1  # includes time channel
    if lead_lag:
        d = 2 * d

    single_sig_dim = sum(d**m for m in range(1, depth + 1))
    num_segments = (2**dyadic_depth) - 1 if dyadic_depth >= 1 else 1
    return num_segments * single_sig_dim


def add_time_channel(path: torch.Tensor, t_span: Tuple[float, float] = (0.0, 1.0)) -> torch.Tensor:
    """Appends a normalized linear time channel [t, x_t] to the path.

    Args:
        path: Tensor of shape [B, L, C] or [L, C]
        t_span: Tuple of (t_start, t_end)

    Returns:
        Tensor with appended time channel of shape [B, L, C + 1] or [L, C + 1]
    """
    is_2d = path.ndim == 2
    if is_2d:
        path = path.unsqueeze(0)

    B, L, C = path.shape
    t = torch.linspace(t_span[0], t_span[1], steps=L, device=path.device, dtype=path.dtype)
    t = t.view(1, L, 1).expand(B, L, 1)

    out = torch.cat([t, path], dim=-1)
    if is_2d:
        out = out.squeeze(0)
    return out


def lead_lag_transform(path: torch.Tensor) -> torch.Tensor:
    """Computes the canonical lead-lag transformation for a batch of paths.

    For a path of shape [B, L, D], transforms it into shape [B, 2L - 1, 2D],
    where lead advances first and lag follows.

    Args:
        path: Tensor of shape [B, L, D] or [L, D]

    Returns:
        Tensor of shape [..., 2L - 1, 2D]
    """
    is_2d = path.ndim == 2
    if is_2d:
        path = path.unsqueeze(0)

    B, L, D = path.shape
    if L <= 1:
        out = torch.cat([path, path], dim=-1)
        return out.squeeze(0) if is_2d else out

    # Construct lead and lag components of length 2L - 1
    # lead: x0, x1, x1, x2, x2, ..., x_{L-1}
    # lag:  x0, x0, x1, x1, x2, ..., x_{L-1}
    lead = torch.repeat_interleave(path, 2, dim=1)[:, 1:]  # [B, 2L - 1, D]
    lag = torch.repeat_interleave(path, 2, dim=1)[:, :-1]  # [B, 2L - 1, D]

    out = torch.cat([lead, lag], dim=-1)  # [B, 2L - 1, 2D]
    if is_2d:
        out = out.squeeze(0)
    return out


def _step_signature(dx: torch.Tensor, depth: int) -> List[torch.Tensor]:
    """Computes tensor powers of single increment dx up to depth:

    s_m = (1 / m!) * dx^(tensor m)
    """
    B, D = dx.shape
    powers = []
    curr = dx  # [B, D]
    powers.append(curr)

    for m in range(2, depth + 1):
        # curr is [B, D^(m-1)], dx is [B, D]
        # outer product per batch: [B, D^(m-1), 1] * [B, 1, D] -> [B, D^m]
        curr = torch.bmm(curr.unsqueeze(2), dx.unsqueeze(1)).view(B, -1)
        curr = curr / m
        powers.append(curr)

    return powers


def _chen_multiply(
    u: List[torch.Tensor],
    v: List[torch.Tensor],
    depth: int,
    dim: int,
) -> List[torch.Tensor]:
    """Applies Chen's relation to multiply two signatures u and v up to depth:

    w_m = u_m + v_m + sum_{j=1}^{m-1} (u_j (x) v_{m-j})
    """
    B = u[0].shape[0]
    w = []
    for m in range(1, depth + 1):
        # Base terms: u_m + v_m
        term = u[m - 1] + v[m - 1]
        # Cross terms
        for j in range(1, m):
            uj = u[j - 1]  # [B, dim^j]
            vm_j = v[m - j - 1]  # [B, dim^(m-j)]
            prod = torch.bmm(uj.unsqueeze(2), vm_j.unsqueeze(1)).view(B, -1)
            term = term + prod
        w.append(term)
    return w


def path_signature(path: torch.Tensor, depth: int = 4) -> torch.Tensor:
    """Computes truncated path signature up to depth using Chen's relation in pure PyTorch.

    Args:
        path: Tensor of shape [B, L, D]
        depth: Truncation depth (default 4)

    Returns:
        Tensor of shape [B, signature_dim] where signature_dim = sum(D^m for m in 1..depth)
    """
    if path.ndim == 2:
        path = path.unsqueeze(0)

    B, L, D = path.shape
    if L < 2:
        total_dim = sum(D**m for m in range(1, depth + 1))
        return torch.zeros(B, total_dim, device=path.device, dtype=path.dtype)

    # Increments: dx shape [L - 1, B, D]
    dx = (path[:, 1:, :] - path[:, :-1, :]).permute(1, 0, 2)  # [L-1, B, D]
    num_steps = dx.shape[0]

    # Initialize running signature with first step
    u = _step_signature(dx[0], depth)

    for i in range(1, num_steps):
        v = _step_signature(dx[i], depth)
        u = _chen_multiply(u, v, depth, D)

    # Concatenate all depth levels: [B, total_dim]
    return torch.cat(u, dim=-1)


def compute_signature_features(
    waveform: torch.Tensor,
    depth: int = 4,
    dyadic_depth: int = 2,
    lead_lag: bool = True,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Extracts signature features from waveform batch, with dyadic decomposition and lead-lag.

    Args:
        waveform: Tensor of shape [B, L, C] or [B, L]
        depth: Signature depth (default 4)
        dyadic_depth: Dyadic level (1 = full path; 2 = full path, first half, second half)
        lead_lag: Whether to apply lead-lag transform
        mean: Optional feature-wise mean tensor for normalization [sig_dim]
        std: Optional feature-wise std tensor for normalization [sig_dim]

    Returns:
        Tensor of signature features [B, sig_dim]
    """
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(-1)

    # 1. Add normalized time channel [t, x_t] across the full waveform
    p_timed = add_time_channel(waveform, t_span=(0.0, 1.0))

    # 2. Lead-lag transform on full path
    p_transformed = lead_lag_transform(p_timed) if lead_lag else p_timed

    # 3. Dyadic decomposition on the transformed path
    sig_full = path_signature(p_transformed, depth=depth)
    sig_parts = [sig_full]

    if dyadic_depth >= 2:
        L_trans = p_transformed.shape[1]
        mid = L_trans // 2
        p_half1 = p_transformed[:, : mid + 1, :]
        p_half2 = p_transformed[:, mid:, :]
        sig_half1 = path_signature(p_half1, depth=depth)
        sig_half2 = path_signature(p_half2, depth=depth)
        sig_parts.extend([sig_half1, sig_half2])

    full_sig = torch.cat(sig_parts, dim=-1)

    # 4. Optional feature normalization
    if mean is not None and std is not None:
        mean = mean.to(device=full_sig.device, dtype=full_sig.dtype)
        std = std.to(device=full_sig.device, dtype=full_sig.dtype)
        full_sig = (full_sig - mean) / (std.clamp_min(1e-5))

    return full_sig

