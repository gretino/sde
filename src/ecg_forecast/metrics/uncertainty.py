from typing import Dict
import numpy as np
import torch


def compute_uncertainty_metrics(
    samples: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """Computes 90% prediction interval coverage and width over multi-sample prior forecasts.
    samples: [Num_Samples, B, 200, num_leads]
    target: [B, 200, num_leads]
    """
    samples_np = samples.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    p5 = np.percentile(samples_np, 5, axis=0)   # [B, 200, num_leads]
    p95 = np.percentile(samples_np, 95, axis=0) # [B, 200, num_leads]

    width = p95 - p5
    mean_width = float(np.mean(width))

    coverage_mask = (target_np >= p5) & (target_np <= p95)
    mean_coverage = float(np.mean(coverage_mask.astype(np.float32)))

    return {
        "coverage_90": mean_coverage,
        "width_90": mean_width,
    }
