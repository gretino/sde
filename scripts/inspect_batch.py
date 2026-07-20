import argparse
import torch
from torch.utils.data import DataLoader

from ecg_forecast.config import load_config
from ecg_forecast.data import ECGWindowDataset, ecg_collate_fn
from ecg_forecast.models import LatentSDEForecaster


def main():
    parser = argparse.ArgumentParser(description="Inspect batch shapes and dataset statistics")
    parser.add_argument("--config", type=str, default="configs/debug_lead2.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"=== Inspecting Dataset with Config: {args.config} ===")
    dataset = ECGWindowDataset(config=cfg.data, split="train")
    print(f"Total dataset items: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=ecg_collate_fn)
    batch = next(iter(loader))

    print("\n--- Batch Tensors & Shapes ---")
    print(f"Record IDs: {batch['record_ids'][:4]}")
    print(f"context_waveform: {batch['context_waveform'].shape} (dtype: {batch['context_waveform'].dtype})")
    print(f"future_waveform:  {batch['future_waveform'].shape} (dtype: {batch['future_waveform'].dtype})")
    print(f"context_times:     {batch['context_times'].shape}")
    print(f"future_times:      {batch['future_times'].shape}")
    print(f"normalization_mean:{batch['normalization_mean'].shape}")
    print(f"normalization_std: {batch['normalization_std'].shape}")
    print(f"future_r_peaks (sample 0): {batch['future_r_peaks'][0]}")

    # Verify context-only normalization: context mean should be ~0 and std ~1 per sample
    c_mean = batch["context_waveform"].mean(dim=1)
    c_std = batch["context_waveform"].std(dim=1)
    print(f"\nContext normalized mean (first 3 samples): {c_mean[:3].squeeze().tolist()}")
    print(f"Context normalized std  (first 3 samples): {c_std[:3].squeeze().tolist()}")

    # Test model forward pass
    print("\n--- Model Forward Pass Inspection ---")
    model = LatentSDEForecaster(config=cfg.model)
    post_out = model.forward_posterior(batch["context_waveform"], batch["future_waveform"])
    prior_out = model.forward_prior(batch["context_waveform"], num_samples=1)

    print(f"Posterior decoded waveform mean: {post_out.waveform_mean.shape}")
    print(f"Posterior decoded scale:         {post_out.waveform_scale.shape} (values: {post_out.waveform_scale.tolist()})")
    print(f"Posterior latent path:           {post_out.latent_path.shape}")
    print(f"Initial KL:                      {post_out.initial_kl.item():.4f}")
    print(f"Path KL:                         {post_out.path_kl.item():.4f}")
    print(f"Prior decoded waveform mean:     {prior_out.waveform_mean.shape}")
    print("\n=== Batch inspection completed successfully! ===")


if __name__ == "__main__":
    main()
