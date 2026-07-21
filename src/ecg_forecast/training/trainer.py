import os
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import Config
from ..models.latent_sde_forecaster import LatentSDEForecaster, ForecastOutput
from ..losses.elbo import compute_elbo_loss
from ..losses.morphology import compute_morphology_loss
from ..losses.schedules import get_loss_weights
from ..metrics.waveform import compute_waveform_metrics
from ..metrics.rhythm import compute_rhythm_metrics
from ..visualization.forecasts import plot_lead2_forecast_panel
from .checkpointing import save_checkpoint, load_checkpoint
from .logging import Logger


class Trainer:
    """3-Stage Trainer for Conditional Latent SDE ECG Forecaster with Multi-GPU support."""

    def __init__(
        self,
        config: Config,
        model: LatentSDEForecaster,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: Optional[str] = None,
        use_wandb: bool = False,
        record_splits: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.raw_model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.record_splits = record_splits

        # Multi-GPU setup
        num_gpus = torch.cuda.device_count()
        if device is None:
            self.device = torch.device("cuda:0" if num_gpus > 0 else "cpu")
        else:
            self.device = torch.device(device)

        self.raw_model.to(self.device)

        if num_gpus > 1 and self.device.type == "cuda":
            print(f"[Trainer] Utilizing {num_gpus} GPUs via DataParallel")
            self.model = nn.DataParallel(self.raw_model)
        else:
            self.model = self.raw_model

        self.logger = Logger(
            use_wandb=use_wandb,
            project_name=config.training.wandb_project,
            run_name=config.training.run_name,
            config=config,
        )

        self.optimizer = torch.optim.AdamW(
            self.raw_model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        self.use_amp = self.config.training.mixed_precision and self.device.type == "cuda"
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.global_step = 0

    def get_unwrapped_model(self) -> LatentSDEForecaster:
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    def train_epoch(
        self,
        stage: str,
        epoch_in_stage: int,
        total_stage_epochs: int,
    ) -> Dict[str, float]:
        self.model.train()
        unwrapped = self.get_unwrapped_model()

        beta_init, beta_path = get_loss_weights(stage, epoch_in_stage, total_stage_epochs)
        epoch_metrics = {
            "loss": [],
            "nll": [],
            "initial_kl": [],
            "path_kl": [],
            "morphology_loss": [],
            "latent_temporal_std": [],
        }

        for batch in tqdm(self.train_loader, desc=f"Train Stage {stage} Epoch {epoch_in_stage+1}", leave=False):
            c_wf = batch["context_waveform"].to(self.device, non_blocking=True)
            f_wf = batch["future_waveform"].to(self.device, non_blocking=True)
            c_times = batch["context_times"].to(self.device, non_blocking=True)
            f_times = batch["future_times"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                out_dict = self.model(
                    context_waveform=c_wf,
                    future_waveform=f_wf,
                    context_times=c_times,
                    future_times=f_times,
                    mode="posterior",
                )
                output = ForecastOutput.from_dict(out_dict)

                elbo_loss, elbo_dict = compute_elbo_loss(
                    pred_mean=output.waveform_mean,
                    target=f_wf,
                    scale=output.waveform_scale,
                    initial_kl=output.initial_kl,
                    path_kl=output.path_kl,
                    beta_initial=beta_init,
                    beta_path=beta_path,
                )

                morph_loss, morph_dict = compute_morphology_loss(
                    pred=output.waveform_mean,
                    target=f_wf,
                    lambda_derivative=self.config.loss.lambda_derivative,
                    lambda_spectral=self.config.loss.lambda_spectral,
                )

                total_loss = elbo_loss + morph_loss

            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.config.training.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.config.training.clip_grad)
                self.optimizer.step()

            self.global_step += 1

            latent_std = output.latent_path.std(dim=1).mean()
            epoch_metrics["loss"].append(float(total_loss.item()))
            epoch_metrics["nll"].append(elbo_dict["nll"])
            epoch_metrics["initial_kl"].append(elbo_dict["initial_kl"])
            epoch_metrics["path_kl"].append(elbo_dict["path_kl"])
            epoch_metrics["morphology_loss"].append(morph_dict["morphology_loss"])
            epoch_metrics["latent_temporal_std"].append(float(latent_std.item()))

        res = {k: (float(np.mean(v)) if len(v) > 0 else 0.0) for k, v in epoch_metrics.items()}
        diffusion = unwrapped.sde.sde_func.sigma
        res["diffusion_mean"] = float(diffusion.mean().item())
        res["diffusion_min"] = float(diffusion.min().item())
        res["diffusion_max"] = float(diffusion.max().item())

        return res

    @torch.no_grad()
    def evaluate(self, save_vis_path: Optional[str] = None) -> Dict[str, float]:
        self.model.eval()
        unwrapped = self.get_unwrapped_model()

        val_metrics = {
            "post_nll": [],
            "prior_nll": [],
            "post_pearson": [],
            "prior_pearson": [],
            "post_rpeak_f1": [],
            "prior_rpeak_f1": [],
            "mse": [],
            "mae": [],
            "zero_peaks": [],
            "total_samples": [],
        }

        vis_sample_done = False

        for batch in self.val_loader:
            c_wf = batch["context_waveform"].to(self.device, non_blocking=True)
            f_wf = batch["future_waveform"].to(self.device, non_blocking=True)
            c_times = batch["context_times"].to(self.device, non_blocking=True)
            f_times = batch["future_times"].to(self.device, non_blocking=True)

            post_dict = self.model(
                context_waveform=c_wf,
                future_waveform=f_wf,
                context_times=c_times,
                future_times=f_times,
                mode="posterior",
            )
            post_out = ForecastOutput.from_dict(post_dict)
            post_elbo, post_dict_loss = compute_elbo_loss(
                post_out.waveform_mean, f_wf, post_out.waveform_scale, post_out.initial_kl, post_out.path_kl
            )
            post_wf_m = compute_waveform_metrics(post_out.waveform_mean, f_wf)
            post_rhythm_m = compute_rhythm_metrics(post_out.waveform_mean, batch["future_r_peaks"])

            prior_dict = self.model(
                context_waveform=c_wf,
                context_times=c_times,
                future_times=f_times,
                mode="prior",
                num_samples=1,
            )
            prior_out = ForecastOutput.from_dict(prior_dict)
            prior_elbo, prior_dict_loss = compute_elbo_loss(
                prior_out.waveform_mean, f_wf, prior_out.waveform_scale, prior_out.initial_kl, prior_out.path_kl
            )
            prior_wf_m = compute_waveform_metrics(prior_out.waveform_mean, f_wf)
            prior_rhythm_m = compute_rhythm_metrics(prior_out.waveform_mean, batch["future_r_peaks"])

            val_metrics["post_nll"].append(post_dict_loss["nll"])
            val_metrics["prior_nll"].append(prior_dict_loss["nll"])
            val_metrics["post_pearson"].append(post_wf_m["pearson"])
            val_metrics["prior_pearson"].append(prior_wf_m["pearson"])
            val_metrics["post_rpeak_f1"].append(post_rhythm_m["rpeak_f1"])
            val_metrics["prior_rpeak_f1"].append(prior_rhythm_m["rpeak_f1"])
            val_metrics["mse"].append(prior_wf_m["mse"])
            val_metrics["mae"].append(prior_wf_m["mae"])
            val_metrics["zero_peaks"].append(prior_rhythm_m["zero_peak_count"])
            val_metrics["total_samples"].append(prior_rhythm_m["total_samples"])

            if save_vis_path is not None and not vis_sample_done:
                vis_sample_done = True
                prior_multi = unwrapped.forward_prior(c_wf[:1], c_times[:1], f_times[:1], num_samples=16)
                c_np = c_wf[0, :, 0].cpu().numpy()
                f_np = f_wf[0, :, 0].cpu().numpy()
                post_np = post_out.waveform_mean[0, :, 0].cpu().numpy()
                prior_samples_np = prior_multi.waveform_mean[:, :, 0].cpu().numpy()

                plot_lead2_forecast_panel(
                    context_wf=c_np,
                    gt_future_wf=f_np,
                    posterior_recon=post_np,
                    prior_samples=prior_samples_np,
                    save_path=save_vis_path,
                )

        tot_zero = int(sum(val_metrics["zero_peaks"]))
        tot_samples = int(sum(val_metrics["total_samples"]))

        res = {k: (float(np.mean(v)) if len(v) > 0 else 0.0) for k, v in val_metrics.items() if k not in ["zero_peaks", "total_samples"]}
        res["zero_peaks_count"] = float(tot_zero)
        res["zero_peaks_pct"] = float(tot_zero / max(1, tot_samples) * 100.0)
        res["prior_forecast_score"] = res["prior_nll"] + (1.0 - res["prior_pearson"]) + (1.0 - res["prior_rpeak_f1"])
        return res

    def run_training(self, resume_path: Optional[str] = None, start_stage: Optional[str] = None):
        ckpt_dir = self.config.training.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        vis_dir = os.path.join(ckpt_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        unwrapped = self.get_unwrapped_model()

        if resume_path is not None:
            ckpt = load_checkpoint(resume_path, model=unwrapped, optimizer=self.optimizer, device=str(self.device))
            saved_stage = ckpt.get("stage", "A")
            saved_epoch = ckpt.get("epoch", 0)
            print(f"\n[Trainer] Successfully loaded checkpoint from {resume_path} (Stage: {saved_stage}, Saved Epoch: {saved_epoch})")
            if start_stage is None:
                # Default resume behavior: if checkpoint was Stage A, resume directly into Stage B
                if saved_stage == "A":
                    start_stage = "B"
                else:
                    start_stage = saved_stage
        
        start_stage = (start_stage or "A").upper()

        epochs_a = self.config.training.posterior_warmup_epochs
        epochs_b = self.config.training.prior_alignment_epochs
        epochs_c = self.config.training.forecast_refinement_epochs
        total_epoch = 0

        # Stage A Loop
        if start_stage == "A":
            print("\n=== Starting Stage A: Posterior Reconstruction Warmup ===")
            best_stage_a_nll = float("inf")
            for epoch in range(epochs_a):
                total_epoch += 1
                train_m = self.train_epoch(stage="A", epoch_in_stage=epoch, total_stage_epochs=epochs_a)
                vis_path = os.path.join(vis_dir, f"stage_A_epoch{epoch+1:02d}.png")
                val_m = self.evaluate(save_vis_path=vis_path)

                step_metrics = {**train_m, **val_m}
                self.logger.log_summary("A", epoch + 1, step_metrics)
                print(f"  [R-peak Detection] {int(val_m['zero_peaks_count'])} zero-peak forecasts out of val set ({val_m['zero_peaks_pct']:.1f}%)")
                self.logger.log(step_metrics, step=total_epoch)
                self.logger.log_image("stage_A_visualization", vis_path, step=total_epoch)

                if val_m["post_nll"] < best_stage_a_nll:
                    best_stage_a_nll = val_m["post_nll"]
                    save_checkpoint(
                        path=os.path.join(ckpt_dir, "posterior_warmup_best.pt"),
                        model=unwrapped,
                        optimizer=self.optimizer,
                        epoch=epoch + 1,
                        stage="A",
                        metrics=val_m,
                        global_step=self.global_step,
                        config=self.config,
                        record_splits=self.record_splits,
                    )
        else:
            print(f"\n[Trainer] Skipping Stage A (Resuming from Stage {start_stage})")
            total_epoch += epochs_a

        # Stage B Loop
        if start_stage in ["A", "B"]:
            print("\n=== Starting Stage B: Prior Alignment ===")
            best_stage_b_score = float("inf")
            for epoch in range(epochs_b):
                total_epoch += 1
                train_m = self.train_epoch(stage="B", epoch_in_stage=epoch, total_stage_epochs=epochs_b)
                vis_path = os.path.join(vis_dir, f"stage_B_epoch{epoch+1:02d}.png")
                val_m = self.evaluate(save_vis_path=vis_path)

                step_metrics = {**train_m, **val_m}
                self.logger.log_summary("B", epoch + 1, step_metrics)
                print(f"  [R-peak Detection] {int(val_m['zero_peaks_count'])} zero-peak forecasts out of val set ({val_m['zero_peaks_pct']:.1f}%)")
                self.logger.log(step_metrics, step=total_epoch)
                self.logger.log_image("stage_B_visualization", vis_path, step=total_epoch)

                if val_m["prior_forecast_score"] < best_stage_b_score:
                    best_stage_b_score = val_m["prior_forecast_score"]
                    save_checkpoint(
                        path=os.path.join(ckpt_dir, "prior_alignment_best.pt"),
                        model=unwrapped,
                        optimizer=self.optimizer,
                        epoch=epoch + 1,
                        stage="B",
                        metrics=val_m,
                        global_step=self.global_step,
                        config=self.config,
                        record_splits=self.record_splits,
                    )
        else:
            total_epoch += epochs_b

        # Stage C Loop
        print("\n=== Starting Stage C: Forecast Refinement ===")
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.config.training.learning_rate * 0.1

        best_stage_c_score = float("inf")
        for epoch in range(epochs_c):
            total_epoch += 1
            train_m = self.train_epoch(stage="C", epoch_in_stage=epoch, total_stage_epochs=epochs_c)
            vis_path = os.path.join(vis_dir, f"stage_C_epoch{epoch+1:02d}.png")
            val_m = self.evaluate(save_vis_path=vis_path)

            step_metrics = {**train_m, **val_m}
            self.logger.log_summary("C", epoch + 1, step_metrics)
            print(f"  [R-peak Detection] {int(val_m['zero_peaks_count'])} zero-peak forecasts out of val set ({val_m['zero_peaks_pct']:.1f}%)")
            self.logger.log(step_metrics, step=total_epoch)
            self.logger.log_image("stage_C_visualization", vis_path, step=total_epoch)

            if val_m["prior_forecast_score"] < best_stage_c_score:
                best_stage_c_score = val_m["prior_forecast_score"]
                save_checkpoint(
                    path=os.path.join(ckpt_dir, "final_best.pt"),
                    model=unwrapped,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    stage="C",
                    metrics=val_m,
                    global_step=self.global_step,
                    config=self.config,
                    record_splits=self.record_splits,
                )

        print(f"\n=== Training Complete. Checkpoints saved to {ckpt_dir} ===")
