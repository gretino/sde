import os
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
import yaml


@dataclass
class DataConfig:
    dataset_name: str = "incart"
    data_dir: str = "data/incart"
    num_leads: int = 12
    lead_indices: Optional[List[int]] = None
    sampling_rate: int = 100
    context_seconds: float = 5.0
    future_seconds: float = 2.0
    stride_seconds: float = 1.0
    split_seed: int = 42
    cache_dir: str = "cache/preprocessed"

    def __init__(
        self,
        dataset_name: str = "incart",
        data_dir: str = "data/incart",
        num_leads: int = 12,
        lead_indices: Optional[List[int]] = None,
        sampling_rate: int = 100,
        context_seconds: Optional[float] = None,
        context_duration: Optional[float] = None,
        future_seconds: Optional[float] = None,
        future_duration: Optional[float] = None,
        stride_seconds: Optional[float] = None,
        stride: Optional[float] = None,
        split_seed: int = 42,
        cache_dir: str = "cache/preprocessed",
        **kwargs,
    ):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.num_leads = num_leads
        self.lead_indices = lead_indices
        self.sampling_rate = sampling_rate

        ctx = context_seconds if context_seconds is not None else (context_duration if context_duration is not None else 5.0)
        fut = future_seconds if future_seconds is not None else (future_duration if future_duration is not None else 2.0)
        std = stride_seconds if stride_seconds is not None else (stride if stride is not None else 1.0)

        self.context_seconds = float(ctx)
        self.future_seconds = float(fut)
        self.stride_seconds = float(std)

        self.split_seed = split_seed
        self.cache_dir = cache_dir

    @property
    def context_duration(self) -> float:
        return self.context_seconds

    @property
    def future_duration(self) -> float:
        return self.future_seconds

    @property
    def stride(self) -> float:
        return self.stride_seconds

    @property
    def context_samples(self) -> int:
        return int(round(self.context_seconds * self.sampling_rate))

    @property
    def future_samples(self) -> int:
        return int(round(self.future_seconds * self.sampling_rate))

    @property
    def total_samples(self) -> int:
        return self.context_samples + self.future_samples


@dataclass
class ModelConfig:
    latent_dim: int = 32
    context_dim: int = 128
    num_leads: int = 12
    dt: float = 0.01
    latent_rate: int = 25

    @property
    def future_latent_steps(self) -> int:
        return int(round(2.0 * self.latent_rate))


@dataclass
class LossConfig:
    beta_initial: float = 1.0
    beta_path: float = 1.0
    lambda_derivative: float = 0.5
    lambda_spectral: float = 0.1
    stage_b_initial_teacher_weight: float = 0.01
    stage_b_drift_teacher_weight: float = 0.01
    stage_c_initial_kl_start: float = 1e-5
    stage_c_initial_kl_max: float = 1e-2
    stage_c_path_kl_start: float = 1e-6
    stage_c_path_kl_max: float = 1e-3
    kl_ramp_epochs: int = 20
    max_weighted_kl_ratio: float = 0.20


@dataclass
class StabilityConfig:
    fixed_diffusion_stage_a_b: float = 0.01
    diffusion_min_stage_c: float = 0.005
    diffusion_max_stage_c: float = 0.05
    fixed_observation_scale_stage_a_b: float = 0.10
    observation_scale_min_stage_c: float = 0.03
    observation_scale_max_stage_c: float = 0.30


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    stage_a_learning_rate: float = 3e-4
    stage_b_prior_learning_rate: float = 3e-4
    stage_b_context_learning_rate: float = 1e-5
    stage_c_prior_learning_rate: float = 1e-4
    stage_c_shared_learning_rate: float = 1e-5
    stage_c_diffusion_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    clip_grad: float = 1.0
    posterior_warmup_epochs: int = 20
    prior_alignment_epochs: int = 50
    forecast_refinement_epochs: int = 20
    num_workers: int = 0
    mixed_precision: bool = False
    prior_samples_eval: int = 16
    use_wandb: bool = True
    wandb_project: str = "ecg-natural-dynamics"
    run_name: Optional[str] = None
    seed: int = 42
    checkpoint_dir: str = "checkpoints/incart_12lead"

    def __init__(
        self,
        batch_size: int = 32,
        learning_rate: Optional[float] = None,
        lr: Optional[float] = None,
        stage_a_learning_rate: Optional[float] = None,
        stage_b_prior_learning_rate: Optional[float] = None,
        stage_b_context_learning_rate: Optional[float] = None,
        stage_c_prior_learning_rate: Optional[float] = None,
        stage_c_shared_learning_rate: Optional[float] = None,
        stage_c_diffusion_learning_rate: Optional[float] = None,
        weight_decay: float = 1e-4,
        clip_grad: float = 1.0,
        posterior_warmup_epochs: Optional[int] = None,
        epochs_stage_a: Optional[int] = None,
        prior_alignment_epochs: Optional[int] = None,
        epochs_stage_b: Optional[int] = None,
        forecast_refinement_epochs: Optional[int] = None,
        epochs_stage_c: Optional[int] = None,
        num_workers: int = 0,
        mixed_precision: bool = False,
        prior_samples_eval: int = 16,
        use_wandb: bool = True,
        wandb_project: str = "ecg-natural-dynamics",
        run_name: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        seed: int = 42,
        checkpoint_dir: str = "checkpoints/incart_12lead",
        **kwargs,
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate if learning_rate is not None else (lr if lr is not None else 3e-4)
        self.stage_a_learning_rate = stage_a_learning_rate if stage_a_learning_rate is not None else self.learning_rate
        self.stage_b_prior_learning_rate = stage_b_prior_learning_rate if stage_b_prior_learning_rate is not None else self.learning_rate
        self.stage_b_context_learning_rate = stage_b_context_learning_rate if stage_b_context_learning_rate is not None else 1e-5
        self.stage_c_prior_learning_rate = stage_c_prior_learning_rate if stage_c_prior_learning_rate is not None else 1e-4
        self.stage_c_shared_learning_rate = stage_c_shared_learning_rate if stage_c_shared_learning_rate is not None else 1e-5
        self.stage_c_diffusion_learning_rate = stage_c_diffusion_learning_rate if stage_c_diffusion_learning_rate is not None else 1e-6

        self.weight_decay = weight_decay
        self.clip_grad = clip_grad

        self.posterior_warmup_epochs = posterior_warmup_epochs if posterior_warmup_epochs is not None else (epochs_stage_a if epochs_stage_a is not None else 20)
        self.prior_alignment_epochs = prior_alignment_epochs if prior_alignment_epochs is not None else (epochs_stage_b if epochs_stage_b is not None else 50)
        self.forecast_refinement_epochs = forecast_refinement_epochs if forecast_refinement_epochs is not None else (epochs_stage_c if epochs_stage_c is not None else 20)

        self.num_workers = num_workers
        self.mixed_precision = mixed_precision
        self.prior_samples_eval = prior_samples_eval
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.run_name = run_name if run_name is not None else wandb_run_name
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir

    @property
    def lr(self) -> float:
        return self.learning_rate

    @property
    def epochs_stage_a(self) -> int:
        return self.posterior_warmup_epochs

    @property
    def epochs_stage_b(self) -> int:
        return self.prior_alignment_epochs

    @property
    def epochs_stage_c(self) -> int:
        return self.forecast_refinement_epochs


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)


def load_config(yaml_path: str) -> Config:
    with open(yaml_path, "r") as f:
        cfg_dict = yaml.safe_load(f) or {}

    default_name = os.path.splitext(os.path.basename(yaml_path))[0]

    data_raw = cfg_dict.get("data", {})
    model_raw = cfg_dict.get("model", {})
    loss_raw = cfg_dict.get("loss", {})
    train_raw = cfg_dict.get("training", {})
    stability_raw = cfg_dict.get("stability", {})

    if not any(k in cfg_dict for k in ["data", "model", "loss", "training", "stability"]):
        data_raw = {
            "dataset_name": cfg_dict.get("dataset_name", "incart"),
            "data_dir": cfg_dict.get("data_dir", "data/incart"),
            "num_leads": len(cfg_dict.get("lead_indices", list(range(12)))) if cfg_dict.get("lead_indices") is not None else 12,
            "lead_indices": cfg_dict.get("lead_indices"),
            "sampling_rate": cfg_dict.get("sampling_rate", 100),
            "context_seconds": cfg_dict.get("context_seconds", 5.0),
            "future_seconds": cfg_dict.get("future_seconds", 2.0),
            "stride_seconds": cfg_dict.get("stride_seconds", 1.0),
            "split_seed": cfg_dict.get("seed", 42),
            "cache_dir": cfg_dict.get("cache_dir", "cache/preprocessed"),
        }
        model_raw = {
            "latent_dim": cfg_dict.get("latent_dim", 32),
            "context_dim": cfg_dict.get("context_dim", 128),
            "num_leads": data_raw["num_leads"],
            "dt": cfg_dict.get("dt", 0.01),
            "latent_rate": cfg_dict.get("latent_rate", 25),
        }
        train_raw = {
            "batch_size": cfg_dict.get("batch_size", 32),
            "learning_rate": cfg_dict.get("learning_rate", 3e-4),
            "weight_decay": cfg_dict.get("weight_decay", 1e-4),
            "clip_grad": cfg_dict.get("clip_grad", 1.0),
            "posterior_warmup_epochs": cfg_dict.get("posterior_warmup_epochs", 20),
            "prior_alignment_epochs": cfg_dict.get("prior_alignment_epochs", 50),
            "forecast_refinement_epochs": cfg_dict.get("forecast_refinement_epochs", 20),
            "num_workers": cfg_dict.get("num_workers", 0),
            "mixed_precision": cfg_dict.get("mixed_precision", False),
            "prior_samples_eval": cfg_dict.get("prior_samples_eval", 16),
            "use_wandb": cfg_dict.get("use_wandb", True),
            "wandb_project": cfg_dict.get("wandb_project", "ecg-natural-dynamics"),
            "run_name": cfg_dict.get("run_name", cfg_dict.get("wandb_run_name", default_name)),
            "seed": cfg_dict.get("seed", 42),
            "checkpoint_dir": cfg_dict.get("checkpoint_dir", "checkpoints/incart_12lead"),
        }

    if "run_name" not in train_raw and "wandb_run_name" not in train_raw:
        train_raw["run_name"] = default_name

    data_cfg = DataConfig(**data_raw)
    model_cfg = ModelConfig(**model_raw)
    loss_cfg = LossConfig(**loss_raw)
    training_cfg = TrainingConfig(**train_raw)
    stability_cfg = StabilityConfig(**stability_raw)

    return Config(
        data=data_cfg,
        model=model_cfg,
        loss=loss_cfg,
        training=training_cfg,
        stability=stability_cfg,
    )
