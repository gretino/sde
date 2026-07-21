from typing import Tuple


def get_loss_weights(
    stage: str,
    epoch_in_stage: int,
    total_stage_epochs: int,
    kl_ramp_epochs: int = 20,
    stage_c_initial_kl_start: float = 1e-5,
    stage_c_initial_kl_max: float = 1e-2,
    stage_c_path_kl_start: float = 1e-6,
    stage_c_path_kl_max: float = 1e-3,
) -> Tuple[float, float]:
    """Returns (beta_initial, beta_path) for current stage and epoch within stage."""
    if stage == "A":
        # Stage A: No KL terms (Section 7.3)
        return 0.0, 0.0

    elif stage == "B":
        # Stage B: Girsanov Path KL remains disabled, alignment via teacher losses (Section 8.10)
        return 0.0, 0.0

    elif stage == "C":
        # Stage C: Ramp beta_initial [1e-5 -> 1e-2] and beta_path [1e-6 -> 1e-3] over kl_ramp_epochs (Section 9.5)
        ramp_epochs = max(1, min(kl_ramp_epochs, total_stage_epochs))
        progress = min(1.0, float(epoch_in_stage) / float(ramp_epochs))
        
        beta_initial = stage_c_initial_kl_start * ((stage_c_initial_kl_max / max(1e-12, stage_c_initial_kl_start)) ** progress)
        beta_path = stage_c_path_kl_start * ((stage_c_path_kl_max / max(1e-12, stage_c_path_kl_start)) ** progress)
        return float(beta_initial), float(beta_path)

    else:
        return 0.0, 0.0
