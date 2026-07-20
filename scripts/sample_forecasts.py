import argparse
import os
import torch
from torch.utils.data import DataLoader

from ecg_forecast.config import load_config
from ecg_forecast.data import ECGWindowDataset, ecg_collate_fn
from ecg_forecast.models import LatentSDEForecaster
from ecg_forecast.training import load_checkpoint
from ecg_forecast.visualization import plot_lead2_forecast_panel, plot_12lead_forecast_page


def main():
    parser = argparse.ArgumentParser(description="Sample multi-future forecasts from model")
    parser.add_argument("--config", type=str, default="configs/incart_12lead.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/incart_12lead/final_best.pt")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default="output/forecast_samples")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = ECGWindowDataset(config=cfg.data, split="test")
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=ecg_collate_fn)

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} does not exist.")
        return

    model = LatentSDEForecaster(config=cfg.model)
    load_checkpoint(args.checkpoint, model=model, device=str(device))
    model.to(device)
    model.eval()

    batch = next(iter(test_loader))
    c_wf = batch["context_waveform"].to(device)
    f_wf = batch["future_waveform"].to(device)

    with torch.no_grad():
        post_out = model.forward_posterior(c_wf, f_wf)

        for i in range(min(4, c_wf.size(0))):
            # Prior forecast 16 samples for sample i
            prior_multi = model.forward_prior(c_wf[i : i + 1], num_samples=args.num_samples)
            c_np = c_wf[i, :, 0].cpu().numpy()
            f_np = f_wf[i, :, 0].cpu().numpy()
            post_np = post_out.waveform_mean[i, :, 0].cpu().numpy()
            samples_np = prior_multi.waveform_mean[:, :, 0].cpu().numpy()

            save_path = os.path.join(args.out_dir, f"sample_{i+1}_lead2.png")
            plot_lead2_forecast_panel(
                context_wf=c_np,
                gt_future_wf=f_np,
                posterior_recon=post_np,
                prior_samples=samples_np,
                save_path=save_path,
                title_suffix=f"(Record: {batch['record_ids'][i]})",
            )
            print(f"Saved forecast sample to {save_path}")

            if cfg.model.num_leads == 12:
                # 12-lead page
                prior_single = model.forward_prior(c_wf[i : i + 1], num_samples=1)
                page_path = os.path.join(args.out_dir, f"sample_{i+1}_12lead.png")
                plot_12lead_forecast_page(
                    context_wf=c_wf[i].cpu().numpy(),
                    gt_future_wf=f_wf[i].cpu().numpy(),
                    prior_mean=prior_single.waveform_mean[0].cpu().numpy(),
                    save_path=page_path,
                )
                print(f"Saved 12-lead page to {page_path}")


if __name__ == "__main__":
    main()
