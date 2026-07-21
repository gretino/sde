import os
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ecg_forecast.config import Config
from ecg_forecast.models import LatentSDEForecaster
from ecg_forecast.training import Trainer


def test_stage_checkpoint_reloading(tmp_path):
    cfg = Config()
    cfg.model.num_leads = 1
    cfg.model.latent_dim = 16
    cfg.model.context_dim = 64
    cfg.training.checkpoint_dir = str(tmp_path)
    cfg.training.posterior_warmup_epochs = 1
    cfg.training.prior_alignment_epochs = 1
    cfg.training.forecast_refinement_epochs = 1

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

    trainer.run_training()

    assert os.path.exists(os.path.join(tmp_path, "posterior_warmup_best.pt")), "Stage A best checkpoint missing"
    assert os.path.exists(os.path.join(tmp_path, "prior_alignment_best.pt")), "Stage B best checkpoint missing"
    assert os.path.exists(os.path.join(tmp_path, "final_best.pt")), "Stage C final best checkpoint missing"
