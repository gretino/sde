import pytest
import torch
from ecg_forecast.config import Config
from ecg_forecast.models.latent_sde_forecaster import LatentSDEForecaster
from ecg_forecast.training.trainer import Trainer


def test_stage_a_rhythm_head_training():
    cfg = Config()
    cfg.loss.lambda_rhythm = 0.5
    model = LatentSDEForecaster(cfg.model)
    trainer = Trainer(model=model, config=cfg)

    b = 2
    c_wf = torch.randn(b, 500, cfg.model.num_leads)
    f_wf = torch.randn(b, 50, cfg.model.num_leads)
    r_labels = torch.randint(0, 2, (b, 50)).float()


    batch = {
        "context_waveform": c_wf,
        "future_waveform": f_wf,
        "rpeak_targets": r_labels,
    }

    # Store initial rhythm head weights
    init_weight = model.rhythm_head[0].weight.clone()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_dict = trainer._train_stage_a_step(batch, optimizer)

    assert "rhythm" in loss_dict, "Rhythm loss missing from Stage A step output!"
    assert loss_dict["rhythm"] >= 0.0

    # Ensure rhythm head received gradient updates
    assert not torch.allclose(model.rhythm_head[0].weight, init_weight), "Rhythm head weights were not updated in Stage A!"
