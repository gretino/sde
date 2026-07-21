import os
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import Config
from ..models.latent_sde_forecaster import LatentSDEForecaster, ForecastOutput
from ..losses.elbo import (
    compute_elbo_loss,
    compute_laplace_nll,
    compute_initial_teacher_loss,
    compute_drift_teacher_loss,
    compute_weighted_kl_ratio,
)
from ..losses.morphology import compute_morphology_loss
from ..losses.schedules import get_loss_weights
from ..metrics.waveform import compute_waveform_metrics
from ..metrics.rhythm import compute_rhythm_metrics
from ..visualization.forecasts import plot_lead2_forecast_panel
from .checkpointing import save_checkpoint, load_checkpoint
from .logging import Logger


class Trainer:
    """Refactored 3-Stage Trainer for Conditional Latent SDE ECG Forecaster matching Stage B Stability Revision Plan."""

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

        self.use_amp = self.config.training.mixed_precision and self.device.type == "cuda"
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.global_step = 0
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def get_unwrapped_model(self) -> LatentSDEForecaster:
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    def set_stage_a_trainable_modules(self):
        """Stage A: Train posterior reconstruction; freeze prior heads, prior drift, diffusion, obs scale."""
        unwrapped = self.get_unwrapped_model()
        unwrapped.set_stage("A")

        # Enable all gradients by default
        for p in unwrapped.parameters():
            p.requires_grad = True

        # Freeze prior components, diffusion, and observation scale
        for p in unwrapped.sde.sde_func.prior_drift_net.parameters():
            p.requires_grad = False
        unwrapped.sde.sde_func.raw_sigma.requires_grad = False
        unwrapped.decoder.raw_obs_log_scale.requires_grad = False

    def set_stage_b_trainable_modules(self):
        """Stage B: Freeze posterior teacher & decoder; train prior heads & prior drift (Sections 8.3 & 8.4)."""
        unwrapped = self.get_unwrapped_model()
        unwrapped.set_stage("B")

        # Freeze everything first
        for p in unwrapped.parameters():
            p.requires_grad = False

        # Unfreeze prior initial-state heads and context attention pool
        for p in unwrapped.context_encoder.attn_net.parameters():
            p.requires_grad = True
        for p in unwrapped.context_encoder.fc_mean.parameters():
            p.requires_grad = True
        for p in unwrapped.context_encoder.fc_logvar.parameters():
            p.requires_grad = True

        # Unfreeze prior drift
        for p in unwrapped.sde.sde_func.prior_drift_net.parameters():
            p.requires_grad = True

    def set_stage_c_trainable_modules(self):
        """Stage C: Joint fine-tuning with all modules unfrozen and stage-specific parameter groups."""
        unwrapped = self.get_unwrapped_model()
        unwrapped.set_stage("C")

        for p in unwrapped.parameters():
            p.requires_grad = True

    def build_stage_a_optimizer(self) -> torch.optim.Optimizer:
        unwrapped = self.get_unwrapped_model()
        self.set_stage_a_trainable_modules()
        trainable = [p for p in unwrapped.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.config.training.stage_a_learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        self.optimizer = optimizer
        return optimizer

    def build_stage_b_optimizer(self) -> torch.optim.Optimizer:
        unwrapped = self.get_unwrapped_model()
        self.set_stage_b_trainable_modules()
        
        prior_params = list(unwrapped.context_encoder.fc_mean.parameters()) + \
                       list(unwrapped.context_encoder.fc_logvar.parameters()) + \
                       list(unwrapped.sde.sde_func.prior_drift_net.parameters())
        attn_params = list(unwrapped.context_encoder.attn_net.parameters())

        param_groups = [
            {"params": [p for p in prior_params if p.requires_grad], "lr": self.config.training.stage_b_prior_learning_rate},
            {"params": [p for p in attn_params if p.requires_grad], "lr": self.config.training.stage_b_context_learning_rate},
        ]
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.training.weight_decay,
        )
        self.optimizer = optimizer
        return optimizer

    def build_stage_c_optimizer(self) -> torch.optim.Optimizer:
        unwrapped = self.get_unwrapped_model()
        self.set_stage_c_trainable_modules()

        prior_params = list(unwrapped.context_encoder.fc_mean.parameters()) + \
                       list(unwrapped.context_encoder.fc_logvar.parameters()) + \
                       list(unwrapped.sde.sde_func.prior_drift_net.parameters())
        
        shared_params = list(unwrapped.context_encoder.conv1.parameters()) + \
                        list(unwrapped.context_encoder.res1.parameters()) + \
                        list(unwrapped.context_encoder.conv2.parameters()) + \
                        list(unwrapped.context_encoder.res2.parameters()) + \
                        list(unwrapped.context_encoder.res3.parameters()) + \
                        list(unwrapped.context_encoder.attn_net.parameters()) + \
                        list(unwrapped.posterior_encoder.parameters()) + \
                        list(unwrapped.sde.sde_func.posterior_drift_net.parameters()) + \
                        list(unwrapped.decoder.net.parameters())

        stability_params = [unwrapped.sde.sde_func.raw_sigma, unwrapped.decoder.raw_obs_log_scale]

        param_groups = [
            {"params": [p for p in prior_params if p.requires_grad], "lr": self.config.training.stage_c_prior_learning_rate},
            {"params": [p for p in shared_params if p.requires_grad], "lr": self.config.training.stage_c_shared_learning_rate},
            {"params": [p for p in stability_params if p.requires_grad], "lr": self.config.training.stage_c_diffusion_learning_rate},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.training.weight_decay,
        )
        self.optimizer = optimizer
        return optimizer

    def train_stage_a_epoch(self, epoch_in_stage: int, total_stage_epochs: int) -> Dict[str, float]:
        self.model.train()
        unwrapped = self.get_unwrapped_model()
        epoch_metrics = {
            "loss": [],
            "nll": [],
            "morphology_loss": [],
            "latent_temporal_std": [],
        }

        for batch in tqdm(self.train_loader, desc=f"Train Stage A Epoch {epoch_in_stage+1}", leave=False):
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

                nll = compute_laplace_nll(output.waveform_mean, f_wf, output.waveform_scale)
                morph_loss, morph_dict = compute_morphology_loss(
                    pred=output.waveform_mean,
                    target=f_wf,
                    lambda_derivative=self.config.loss.lambda_derivative,
                    lambda_spectral=self.config.loss.lambda_spectral,
                )
                total_loss = nll + morph_loss

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
            epoch_metrics["nll"].append(float(nll.item()))
            epoch_metrics["morphology_loss"].append(morph_dict["morphology_loss"])
            epoch_metrics["latent_temporal_std"].append(float(latent_std.item()))

        res = {k: (float(np.mean(v)) if len(v) > 0 else 0.0) for k, v in epoch_metrics.items()}
        diffusion = unwrapped.sde.sde_func.sigma
        res["diffusion_mean"] = float(diffusion.mean().item())
        res["diffusion_min"] = float(diffusion.min().item())
        res["diffusion_max"] = float(diffusion.max().item())
        res["observation_scale"] = float(unwrapped.decoder.observation_scale.mean().item())
        return res

    def train_stage_b_epoch(self, epoch_in_stage: int, total_stage_epochs: int) -> Dict[str, float]:
        self.model.train()
        unwrapped = self.get_unwrapped_model()
        epoch_metrics = {
            "loss": [],
            "prior_nll": [],
            "prior_morphology_loss": [],
            "trajectory_loss": [],
            "initial_mean_loss": [],
            "drift_teacher_loss": [],
            "prior_latent_temporal_std": [],
        }

        w_traj = getattr(self.config.loss, "lambda_trajectory", 1.0)
        w_z0 = getattr(self.config.loss, "lambda_z0", 0.1)
        w_drift = getattr(self.config.loss, "lambda_drift", 0.01)

        for batch in tqdm(self.train_loader, desc=f"Train Stage B Epoch {epoch_in_stage+1}", leave=False):
            c_wf = batch["context_waveform"].to(self.device, non_blocking=True)
            f_wf = batch["future_waveform"].to(self.device, non_blocking=True)
            c_times = batch["context_times"].to(self.device, non_blocking=True)
            f_times = batch["future_times"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # 1. Run Detached Posterior Teacher Pass in Deterministic Mode (mu_q, zero diffusion)
            with torch.no_grad():
                c_summary_t, _, _, _ = unwrapped.context_encoder(c_wf)
                full_wf = torch.cat([c_wf, f_wf], dim=1)
                post_summary, rec_path, post_mean_det, post_logvar_det = unwrapped.posterior_encoder(full_wf, c_summary_t)

                ts = f_times[0, ::4] if f_times.dim() == 2 else f_times[::4]

                # Force zero diffusion for deterministic teacher path
                raw_sigma_orig = unwrapped.sde.sde_func.raw_sigma.data.clone()
                unwrapped.sde.sde_func.raw_sigma.data.fill_(-100.0)

                post_latent_path, _ = unwrapped.sde.integrate(
                    z0=post_mean_det, ts=ts, context_summary=c_summary_t, recognition_path=rec_path, mode="posterior"
                )

                num_steps = ts.size(0)
                post_drifts = []
                for k in range(num_steps):
                    tk = ts[k]
                    zk = post_latent_path[:, k, :]
                    post_drifts.append(unwrapped.sde.sde_func.f(tk, zk))
                post_drifts_det = torch.stack(post_drifts, dim=1).detach()

                unwrapped.sde.sde_func.raw_sigma.data.copy_(raw_sigma_orig)

            # 2. Run Deterministic Prior Student Pass (z0 = mu_p, zero diffusion)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                c_summary, c_tokens, prior_mean, prior_logvar = unwrapped.context_encoder(c_wf)

                # Deterministic integration
                unwrapped.sde.sde_func.raw_sigma.data.fill_(-100.0)
                prior_latent_path, _ = unwrapped.sde.integrate(
                    z0=prior_mean, ts=ts, context_summary=c_summary, mode="prior"
                )
                unwrapped.sde.sde_func.raw_sigma.data.copy_(raw_sigma_orig)

                prior_wf_mean, prior_scale = unwrapped.decoder(prior_latent_path, c_summary, target_len=f_wf.size(1))


                # Prior drift on detached teacher states
                prior_drifts_on_teacher = []
                for k in range(num_steps):
                    tk = ts[k]
                    zk_teacher = post_latent_path[:, k, :].detach()
                    prior_drifts_on_teacher.append(unwrapped.sde.sde_func.h(tk, zk_teacher))
                prior_drifts_det = torch.stack(prior_drifts_on_teacher, dim=1)

                # Losses
                prior_nll = compute_laplace_nll(prior_wf_mean, f_wf, prior_scale)
                prior_morph, prior_morph_dict = compute_morphology_loss(
                    pred=prior_wf_mean,
                    target=f_wf,
                    lambda_derivative=self.config.loss.lambda_derivative,
                    lambda_spectral=self.config.loss.lambda_spectral,
                )
                prior_waveform_loss = prior_nll + prior_morph

                # Optional R-peak timing rhythm supervision (Section 10)
                if "future_r_peaks" in batch and len(batch["future_r_peaks"]) > 0:
                    from ..losses.morphology import compute_rhythm_loss
                    pred_r_prob = unwrapped.predict_r_peak_probability(prior_latent_path)
                    rhythm_loss = compute_rhythm_loss(pred_r_prob, batch["future_r_peaks"])
                    w_rhythm = getattr(self.config.loss, "lambda_rhythm", 0.5)
                    prior_waveform_loss = prior_waveform_loss + w_rhythm * rhythm_loss

                from ..losses.elbo import compute_autonomous_trajectory_loss, compute_initial_mean_loss
                traj_loss = compute_autonomous_trajectory_loss(prior_latent_path, post_latent_path)
                z0_loss = compute_initial_mean_loss(prior_mean, post_mean_det)
                drift_teacher_loss = compute_drift_teacher_loss(prior_drifts_det, post_drifts_det)


                total_loss = prior_waveform_loss + w_traj * traj_loss + w_z0 * z0_loss + w_drift * drift_teacher_loss

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
            latent_std = prior_latent_path.std(dim=1).mean()

            epoch_metrics["loss"].append(float(total_loss.item()))
            epoch_metrics["prior_nll"].append(float(prior_nll.item()))
            epoch_metrics["prior_morphology_loss"].append(prior_morph_dict["morphology_loss"])
            epoch_metrics["trajectory_loss"].append(float(traj_loss.item()))
            epoch_metrics["initial_mean_loss"].append(float(z0_loss.item()))
            epoch_metrics["drift_teacher_loss"].append(float(drift_teacher_loss.item()))
            epoch_metrics["prior_latent_temporal_std"].append(float(latent_std.item()))

        res = {k: (float(np.mean(v)) if len(v) > 0 else 0.0) for k, v in epoch_metrics.items()}
        diffusion = unwrapped.sde.sde_func.sigma
        res["diffusion_mean"] = float(diffusion.mean().item())
        res["diffusion_min"] = float(diffusion.min().item())
        res["diffusion_max"] = float(diffusion.max().item())
        res["observation_scale"] = float(unwrapped.decoder.observation_scale.mean().item())
        return res

    def train_stage_c_epoch(self, epoch_in_stage: int, total_stage_epochs: int) -> Dict[str, float]:
        self.model.train()
        unwrapped = self.get_unwrapped_model()

        beta_init, beta_path = get_loss_weights(
            stage="C",
            epoch_in_stage=epoch_in_stage,
            total_stage_epochs=total_stage_epochs,
            kl_ramp_epochs=self.config.loss.kl_ramp_epochs,
            stage_c_initial_kl_start=self.config.loss.stage_c_initial_kl_start,
            stage_c_initial_kl_max=self.config.loss.stage_c_initial_kl_max,
            stage_c_path_kl_start=self.config.loss.stage_c_path_kl_start,
            stage_c_path_kl_max=self.config.loss.stage_c_path_kl_max,
        )

        epoch_metrics = {
            "loss": [],
            "posterior_nll": [],
            "prior_nll": [],
            "initial_kl": [],
            "path_kl": [],
            "weighted_initial_kl": [],
            "weighted_path_kl": [],
            "weighted_kl_ratio": [],
        }

        for batch in tqdm(self.train_loader, desc=f"Train Stage C Epoch {epoch_in_stage+1}", leave=False):
            c_wf = batch["context_waveform"].to(self.device, non_blocking=True)
            f_wf = batch["future_waveform"].to(self.device, non_blocking=True)
            c_times = batch["context_times"].to(self.device, non_blocking=True)
            f_times = batch["future_times"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                # Posterior pass
                post_dict = self.model(
                    context_waveform=c_wf,
                    future_waveform=f_wf,
                    context_times=c_times,
                    future_times=f_times,
                    mode="posterior",
                )
                post_out = ForecastOutput.from_dict(post_dict)

                post_nll = compute_laplace_nll(post_out.waveform_mean, f_wf, post_out.waveform_scale)
                post_morph, _ = compute_morphology_loss(
                    pred=post_out.waveform_mean,
                    target=f_wf,
                    lambda_derivative=self.config.loss.lambda_derivative,
                    lambda_spectral=self.config.loss.lambda_spectral,
                )
                post_wf_loss = post_nll + post_morph

                # Prior stochastic pass
                prior_dict = self.model(
                    context_waveform=c_wf,
                    context_times=c_times,
                    future_times=f_times,
                    mode="prior",
                )
                prior_out = ForecastOutput.from_dict(prior_dict)
                prior_nll = compute_laplace_nll(prior_out.waveform_mean, f_wf, prior_out.waveform_scale)
                prior_morph, _ = compute_morphology_loss(
                    pred=prior_out.waveform_mean,
                    target=f_wf,
                    lambda_derivative=self.config.loss.lambda_derivative,
                    lambda_spectral=self.config.loss.lambda_spectral,
                )
                prior_wf_loss = prior_nll + prior_morph

                # Prior deterministic mean anchor pass (Section 13)
                c_summary, _, prior_mean, _ = unwrapped.context_encoder(c_wf)
                ts = f_times[0, ::4] if f_times.dim() == 2 else f_times[::4]

                raw_sigma_orig = unwrapped.sde.sde_func.raw_sigma.data.clone()
                unwrapped.sde.sde_func.raw_sigma.data.fill_(-100.0)
                mean_latent, _ = unwrapped.sde.integrate(z0=prior_mean, ts=ts, context_summary=c_summary, mode="prior")
                unwrapped.sde.sde_func.raw_sigma.data.copy_(raw_sigma_orig)

                mean_wf, mean_scale = unwrapped.decoder(mean_latent, c_summary)
                mean_nll = compute_laplace_nll(mean_wf, f_wf, mean_scale)
                mean_morph, _ = compute_morphology_loss(mean_wf, f_wf)
                prior_mean_anchor_loss = mean_nll + mean_morph

                init_kl = post_out.initial_kl.mean()
                path_kl = post_out.path_kl.mean()

                w_init_kl = beta_init * init_kl
                w_path_kl = beta_path * path_kl
                waveform_objective = post_wf_loss + prior_wf_loss + prior_mean_anchor_loss
                kl_ratio = compute_weighted_kl_ratio(float(w_init_kl.item()), float(w_path_kl.item()), float(waveform_objective.item()))

                # Enforce max_weighted_kl_ratio <= 0.20 (Section 9.5 & 11.5)
                if kl_ratio > self.config.loss.max_weighted_kl_ratio:
                    w_init_kl = w_init_kl * (self.config.loss.max_weighted_kl_ratio / max(1e-8, kl_ratio))
                    w_path_kl = w_path_kl * (self.config.loss.max_weighted_kl_ratio / max(1e-8, kl_ratio))

                total_loss = post_wf_loss + prior_wf_loss + 0.5 * prior_mean_anchor_loss + w_init_kl + w_path_kl


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

            epoch_metrics["loss"].append(float(total_loss.item()))
            epoch_metrics["posterior_nll"].append(float(post_nll.item()))
            epoch_metrics["prior_nll"].append(float(prior_nll.item()))
            epoch_metrics["initial_kl"].append(float(init_kl.item()))
            epoch_metrics["path_kl"].append(float(path_kl.item()))
            epoch_metrics["weighted_initial_kl"].append(float(w_init_kl.item()))
            epoch_metrics["weighted_path_kl"].append(float(w_path_kl.item()))
            epoch_metrics["weighted_kl_ratio"].append(kl_ratio)

        res = {k: (float(np.mean(v)) if len(v) > 0 else 0.0) for k, v in epoch_metrics.items()}
        diffusion = unwrapped.sde.sde_func.sigma
        res["diffusion_mean"] = float(diffusion.mean().item())
        res["diffusion_min"] = float(diffusion.min().item())
        res["diffusion_max"] = float(diffusion.max().item())
        res["observation_scale"] = float(unwrapped.decoder.observation_scale.mean().item())
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
            "post_latent_temporal_std": [],
            "prior_latent_temporal_std": [],
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
            _, post_dict_loss = compute_elbo_loss(
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
            _, prior_dict_loss = compute_elbo_loss(
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
            val_metrics["post_latent_temporal_std"].append(float(post_out.latent_path.std(dim=1).mean().item()))
            val_metrics["prior_latent_temporal_std"].append(float(prior_out.latent_path.std(dim=1).mean().item()))
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

    def check_collapse_warnings(self, val_m: Dict[str, float]):
        """Section 11.5 Collapse Warnings."""
        if val_m.get("zero_peaks_pct", 0.0) > 50.0:
            print(f"  ⚠️ [WARNING] Zero R-peak percentage exceeds 50%: {val_m['zero_peaks_pct']:.1f}%")
        if val_m.get("weighted_kl_ratio", 0.0) > self.config.loss.max_weighted_kl_ratio:
            print(f"  ⚠️ [WARNING] Weighted KL ratio exceeds limit {self.config.loss.max_weighted_kl_ratio}: {val_m['weighted_kl_ratio']:.4f}")

    def run_training(self, resume_path: Optional[str] = None, start_stage: Optional[str] = None):
        ckpt_dir = self.config.training.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        vis_dir = os.path.join(ckpt_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        unwrapped = self.get_unwrapped_model()

        if resume_path is not None:
            ckpt = load_checkpoint(resume_path, model=unwrapped, device=str(self.device))
            saved_stage = ckpt.get("stage", "A")
            saved_epoch = ckpt.get("epoch", 0)
            print(f"\n[Trainer] Successfully loaded checkpoint from {resume_path} (Saved Stage: {saved_stage}, Saved Epoch: {saved_epoch})")
            if start_stage is None:
                start_stage = "B" if saved_stage == "A" else saved_stage

        start_stage = (start_stage or "A").upper()

        epochs_a = self.config.training.posterior_warmup_epochs
        epochs_b = self.config.training.prior_alignment_epochs
        epochs_c = self.config.training.forecast_refinement_epochs
        total_epoch = 0

        # Stage A Loop
        if start_stage == "A":
            print("\n=== Starting Stage A: Posterior Reconstruction Warmup ===")
            self.optimizer = self.build_stage_a_optimizer()
            best_stage_a_nll = float("inf")

            for epoch in range(epochs_a):
                total_epoch += 1
                train_m = self.train_stage_a_epoch(epoch_in_stage=epoch, total_stage_epochs=epochs_a)
                vis_path = os.path.join(vis_dir, f"stage_A_epoch{epoch+1:02d}.png")
                val_m = self.evaluate(save_vis_path=vis_path)

                step_metrics = {**train_m, **val_m}
                self.logger.log_summary("A", epoch + 1, step_metrics)
                print(f"  [R-peak Detection] {int(val_m['zero_peaks_count'])} zero-peak forecasts out of val set ({val_m['zero_peaks_pct']:.1f}%)")
                self.check_collapse_warnings(val_m)
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
            print(f"\n[Trainer] Skipping Stage A (Starting from Stage {start_stage})")
            total_epoch += epochs_a

        # Stage B Loop
        if start_stage in ["A", "B"]:
            stage_a_ckpt = os.path.join(ckpt_dir, "posterior_warmup_best.pt")
            if os.path.exists(stage_a_ckpt) and start_stage == "B":
                print(f"\n[Trainer] Section 8.2: Reloading best Stage A checkpoint: {stage_a_ckpt}")
                load_checkpoint(stage_a_ckpt, model=unwrapped, device=str(self.device))

            print("\n=== Starting Stage B: Prior Alignment (Teacher-Student Training) ===")
            self.optimizer = self.build_stage_b_optimizer()
            best_stage_b_score = float("inf")

            for epoch in range(epochs_b):
                total_epoch += 1
                train_m = self.train_stage_b_epoch(epoch_in_stage=epoch, total_stage_epochs=epochs_b)
                vis_path = os.path.join(vis_dir, f"stage_B_epoch{epoch+1:02d}.png")
                val_m = self.evaluate(save_vis_path=vis_path)

                step_metrics = {**train_m, **val_m}
                self.logger.log_summary("B", epoch + 1, step_metrics)
                print(f"  [R-peak Detection] {int(val_m['zero_peaks_count'])} zero-peak forecasts out of val set ({val_m['zero_peaks_pct']:.1f}%)")
                self.check_collapse_warnings(val_m)
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
        stage_b_ckpt = os.path.join(ckpt_dir, "prior_alignment_best.pt")
        if os.path.exists(stage_b_ckpt) and start_stage == "C":
            print(f"\n[Trainer] Section 9.2: Reloading best Stage B checkpoint: {stage_b_ckpt}")
            load_checkpoint(stage_b_ckpt, model=unwrapped, device=str(self.device))

        print("\n=== Starting Stage C: Forecast Refinement (Joint Fine-Tuning) ===")
        self.optimizer = self.build_stage_c_optimizer()
        best_stage_c_score = float("inf")

        for epoch in range(epochs_c):
            total_epoch += 1
            train_m = self.train_stage_c_epoch(epoch_in_stage=epoch, total_stage_epochs=epochs_c)
            vis_path = os.path.join(vis_dir, f"stage_C_epoch{epoch+1:02d}.png")
            val_m = self.evaluate(save_vis_path=vis_path)

            step_metrics = {**train_m, **val_m}
            self.logger.log_summary("C", epoch + 1, step_metrics)
            print(f"  [R-peak Detection] {int(val_m['zero_peaks_count'])} zero-peak forecasts out of val set ({val_m['zero_peaks_pct']:.1f}%)")
            self.check_collapse_warnings(val_m)
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
