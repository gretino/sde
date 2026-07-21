import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ecg_forecast.config import Config
from ecg_forecast.models import LatentSDEForecaster
from ecg_forecast.training import Trainer


def test_stage_b_freezing_gradients():
    cfg = Config()
    cfg.model.num_leads = 1
    cfg.model.latent_dim = 16
    cfg.model.context_dim = 64

    model = LatentSDEForecaster(config=cfg.model)

    c_wf = torch.randn(2, 500, 1)
    f_wf = torch.randn(2, 200, 1)
    c_t = torch.linspace(0, 5, 500)
    f_t = torch.linspace(5, 7, 200)

    dataset = TensorDataset(c_wf, f_wf)
    def collate(batch):
        return {
            "context_waveform": torch.stack([b[0] for b in batch]),
            "future_waveform": torch.stack([b[1] for b in batch]),
            "context_times": c_t.unsqueeze(0).expand(len(batch), -1),
            "future_times": f_t.unsqueeze(0).expand(len(batch), -1),
            "future_r_peaks": [torch.tensor([50, 150]) for _ in batch],
        }

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate)
    trainer = Trainer(config=cfg, model=model, train_loader=loader, val_loader=loader, use_wandb=False)

    trainer.build_stage_b_optimizer()
    trainer.train_stage_b_epoch(epoch_in_stage=0, total_stage_epochs=1)

    unwrapped = trainer.get_unwrapped_model()

    # Frozen modules must have no gradients (Section 13.1)
    for p in unwrapped.posterior_encoder.parameters():
        assert p.grad is None or (p.grad == 0).all(), "Posterior encoder should have zero grad in Stage B"
    for p in unwrapped.sde.sde_func.posterior_drift_net.parameters():
        assert p.grad is None or (p.grad == 0).all(), "Posterior drift should have zero grad in Stage B"
    for p in unwrapped.decoder.parameters():
        assert p.grad is None or (p.grad == 0).all(), "Decoder should have zero grad in Stage B"

    # Trainable prior modules must have non-zero gradients
    prior_drift_grad = unwrapped.sde.sde_func.prior_drift_net[0].weight.grad
    assert prior_drift_grad is not None and torch.norm(prior_drift_grad) > 0.0, "Prior drift must have non-zero grad in Stage B"
    prior_mean_grad = unwrapped.context_encoder.fc_mean.weight.grad
    assert prior_mean_grad is not None and torch.norm(prior_mean_grad) > 0.0, "Prior mean head must have non-zero grad in Stage B"
