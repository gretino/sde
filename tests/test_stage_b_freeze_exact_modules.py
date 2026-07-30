import pytest
import torch
from ecg_forecast.config import Config
from ecg_forecast.models.latent_sde_forecaster import LatentSDEForecaster
from ecg_forecast.training.trainer import Trainer


def test_stage_b_freeze_exact_modules():
    cfg = Config()
    model = LatentSDEForecaster(cfg.model)
    trainer = Trainer(model=model, config=cfg)

    trainer._setup_stage_b_freezing()

    # Must be trainable
    for name, p in model.context_encoder.fc_mean.named_parameters():
        assert p.requires_grad, f"fc_mean parameter {name} should be unfrozen!"

    for name, p in model.sde.sde_func.prior_drift_net.named_parameters():
        assert p.requires_grad, f"prior_drift_net parameter {name} should be unfrozen!"

    # Must be frozen
    for name, p in model.posterior_encoder.named_parameters():
        assert not p.requires_grad, f"posterior_encoder parameter {name} should be frozen!"

    for name, p in model.decoder.named_parameters():
        assert not p.requires_grad, f"decoder parameter {name} should be frozen!"

    for name, p in model.context_encoder.attn_net.named_parameters():
        assert not p.requires_grad, f"context_encoder.attn_net parameter {name} should be frozen!"

    for name, p in model.context_encoder.fc_logvar.named_parameters():
        assert not p.requires_grad, f"context_encoder.fc_logvar parameter {name} should be frozen!"
