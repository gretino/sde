import pytest
from ecg_forecast.config import Config
from ecg_forecast.models.latent_sde_forecaster import LatentSDEForecaster
from ecg_forecast.training.trainer import Trainer


def test_stage_b_rhythm_head_frozen():
    cfg = Config()
    model = LatentSDEForecaster(cfg.model)
    trainer = Trainer(model=model, config=cfg)

    trainer._setup_stage_b_freezing()

    for name, p in model.rhythm_head.named_parameters():
        assert not p.requires_grad, f"rhythm_head parameter {name} was unexpectedly unfrozen in Stage B!"
