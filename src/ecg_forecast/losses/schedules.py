from typing import Tuple


def get_loss_weights(
    stage: str,
    epoch_in_stage: int,
    total_stage_epochs: int,
) -> Tuple[float, float]:
    """Returns (beta_initial, beta_path) for current stage and epoch within stage."""
    if stage == "A":
        return 0.0, 0.0

    elif stage == "B":
        # Warmup over first 10 epochs (or half of total_stage_epochs if shorter than 10)
        ramp_epochs = min(10, total_stage_epochs)
        if epoch_in_stage < ramp_epochs:
            weight = float(epoch_in_stage + 1) / float(ramp_epochs)
            return weight, weight
        else:
            return 1.0, 1.0

    elif stage == "C":
        return 1.0, 1.0

    else:
        return 1.0, 1.0
