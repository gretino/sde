import warnings
from typing import Dict
import numpy as np
import torch


def compute_waveform_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Computes MSE, MAE, and Pearson correlation coefficient between predicted and target waveforms safely."""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    if pred_np.size == 0 or target_np.size == 0:
        return {"mse": 0.0, "mae": 0.0, "pearson": 0.0}

    mse = float(np.mean((pred_np - target_np) ** 2))
    mae = float(np.mean(np.abs(pred_np - target_np)))

    b, time_len, num_leads = pred_np.shape
    corrs = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i in range(b):
            for l in range(num_leads):
                p = pred_np[i, :, l]
                t = target_np[i, :, l]
                p_std = float(np.std(p))
                t_std = float(np.std(t))

                if p_std > 1e-6 and t_std > 1e-6:
                    r = np.corrcoef(p, t)[0, 1]
                    if not np.isnan(r):
                        corrs.append(float(r))
                    else:
                        corrs.append(0.0)
                else:
                    corrs.append(0.0)

    pearson = float(np.mean(corrs)) if len(corrs) > 0 else 0.0

    return {
        "mse": mse,
        "mae": mae,
        "pearson": pearson,
    }
