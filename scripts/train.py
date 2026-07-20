import argparse
import os
import torch
from torch.utils.data import DataLoader

from ecg_forecast.config import load_config
from ecg_forecast.data import ECGWindowDataset, ecg_collate_fn
from ecg_forecast.models import LatentSDEForecaster
from ecg_forecast.training import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Conditional Latent SDE ECG Forecaster")
    parser.add_argument("--config", type=str, default="configs/incart_12lead.yaml", help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="Override num workers")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.training.num_workers = args.num_workers

    use_wandb = cfg.training.use_wandb if not args.no_wandb else False
    torch.manual_seed(cfg.training.seed)

    print(f"Loading datasets with config: {args.config}")
    train_dataset = ECGWindowDataset(config=cfg.data, split="train")
    val_dataset = ECGWindowDataset(config=cfg.data, split="val")

    num_gpus = torch.cuda.device_count()
    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=ecg_collate_fn,
        num_workers=cfg.training.num_workers,
        pin_memory=pin_mem,
        persistent_workers=(cfg.training.num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=ecg_collate_fn,
        num_workers=cfg.training.num_workers,
        pin_memory=pin_mem,
        persistent_workers=(cfg.training.num_workers > 0),
    )

    print(f"GPUs Available: {num_gpus} | Batch Size: {cfg.training.batch_size} | Num Workers: {cfg.training.num_workers}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    model = LatentSDEForecaster(config=cfg.model)
    trainer = Trainer(
        config=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        use_wandb=use_wandb,
    )

    trainer.run_training()


if __name__ == "__main__":
    main()
