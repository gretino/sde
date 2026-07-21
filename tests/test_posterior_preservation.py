import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ecg_forecast.config import Config
from ecg_forecast.models import LatentSDEForecaster
from ecg_forecast.training import Trainer


def test_posterior_preservation_during_stage_b():
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

    c_wf_dev = c_wf.to(trainer.device)
    f_wf_dev = f_wf.to(trainer.device)

    # Initial posterior encoding output
    with torch.no_grad():
        c_summary_1, _, _, _ = trainer.get_unwrapped_model().context_encoder(c_wf_dev)
        full_wf = torch.cat([c_wf_dev, f_wf_dev], dim=1)
        _, _, post_m_1, post_logvar_1 = trainer.get_unwrapped_model().posterior_encoder(full_wf, c_summary_1)

    # Run Stage B training steps
    trainer.build_stage_b_optimizer()
    trainer.train_stage_b_epoch(epoch_in_stage=0, total_stage_epochs=1)

    # Posterior encoding output after Stage B training steps
    with torch.no_grad():
        c_summary_2, _, _, _ = trainer.get_unwrapped_model().context_encoder(c_wf_dev)
        _, _, post_m_2, post_logvar_2 = trainer.get_unwrapped_model().posterior_encoder(full_wf, c_summary_2)

    # Verify posterior distribution parameters are NUMERICALLY UNCHANGED (Section 13.4)
    diff = torch.abs(post_m_1 - post_m_2).max().item()
    assert diff < 1e-6, f"Posterior encoder mean changed during Stage B: max diff = {diff}"
