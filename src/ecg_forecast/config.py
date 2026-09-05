import os
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
import yaml


@dataclass
class DataConfig:
    dataset_name: str = "incart"
    data_dir: str = "data/incart"
    leads: List[int] = field(default_factory=lambda: [1])
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
        leads: Optional[List[int]] = None,
        lead_indices: Optional[List[int]] = None,
        num_leads: Optional[int] = None,
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

        if leads is not None:
            self.leads = list(leads)
        elif lead_indices is not None:
            self.leads = list(lead_indices)
        elif num_leads is not None and num_leads != 12:
            self.leads = list(range(num_leads))
        else:
            self.leads = [1]

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
    def lead_indices(self) -> List[int]:
        return self.leads

    @property
    def num_leads(self) -> int:
        return len(self.leads)

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
class SignatureConfig:
    depth: int = 4
    dyadic_depth: int = 2
    lead_lag: bool = True
    normalize: bool = True
    normalize_features: bool = True
    ridge_alpha: float = 0.1
    signatures_dir: str = "artifacts/signatures"

    def __init__(
        self,
        depth: int = 4,
        dyadic_depth: int = 2,
        lead_lag: bool = True,
        normalize: bool = True,
        normalize_features: Optional[bool] = None,
        ridge_alpha: float = 0.1,
        signatures_dir: str = "artifacts/signatures",
        **kwargs,
    ):
        self.depth = depth
        self.dyadic_depth = dyadic_depth
        self.lead_lag = lead_lag
        self.normalize = normalize
        self.normalize_features = normalize_features if normalize_features is not None else normalize
        self.ridge_alpha = ridge_alpha
        self.signatures_dir = signatures_dir


@dataclass
class ModelConfig:
    context_dim: int = 64
    initial_noise_dim: int = 16
    latent_dim: int = 64
    drift_hidden: List[int] = field(default_factory=lambda: [128, 128, 128])
    diffusion_hidden: List[int] = field(default_factory=lambda: [128, 128, 128])
    sigma_min: float = 0.005
    sigma_max: float = 0.20
    num_leads: int = 1

    def __init__(
        self,
        context_dim: int = 64,
        initial_noise_dim: int = 16,
        latent_dim: int = 64,
        drift_hidden: Optional[List[int]] = None,
        diffusion_hidden: Optional[List[int]] = None,
        sigma_min: float = 0.005,
        sigma_max: float = 0.20,
        num_leads: int = 1,
        **kwargs,
    ):
        self.context_dim = context_dim
        self.initial_noise_dim = initial_noise_dim
        self.latent_dim = latent_dim
        self.drift_hidden = list(drift_hidden) if drift_hidden is not None else [128, 128, 128]
        self.diffusion_hidden = list(diffusion_hidden) if diffusion_hidden is not None else [128, 128, 128]
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_leads = num_leads


@dataclass
class SDEConfig:
    type: str = "stratonovich"
    noise_type: str = "diagonal"
    method: str = "reversible_heun"
    adjoint_method: str = "adjoint_reversible_heun"
    dt: float = 0.01

    def __init__(
        self,
        type: str = "stratonovich",
        noise_type: str = "diagonal",
        method: str = "reversible_heun",
        adjoint_method: str = "adjoint_reversible_heun",
        dt: float = 0.01,
        **kwargs,
    ):
        self.type = type
        self.noise_type = noise_type
        self.method = method
        self.adjoint_method = adjoint_method
        self.dt = dt


@dataclass
class TrainingConfig:
    batch_size: int = 64
    num_samples: int = 8
    monte_carlo_samples: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    grad_clip: float = 1.0
    mixed_precision: bool = False
    epochs: int = 50
    seed: int = 42
    num_workers: int = 0
    use_wandb: bool = False
    wandb_project: str = "cnsde-ecg-forecasting"
    run_name: Optional[str] = None
    checkpoint_dir: str = "checkpoints/lead2_cnsde"

    def __init__(
        self,
        batch_size: int = 64,
        num_samples: Optional[int] = None,
        monte_carlo_samples: Optional[int] = None,
        learning_rate: Optional[float] = None,
        lr: Optional[float] = None,
        weight_decay: float = 0.0001,
        grad_clip: float = 1.0,
        clip_grad: Optional[float] = None,
        mixed_precision: bool = False,
        epochs: int = 50,
        seed: int = 42,
        num_workers: int = 0,
        use_wandb: bool = False,
        wandb_project: str = "cnsde-ecg-forecasting",
        run_name: Optional[str] = None,
        checkpoint_dir: str = "checkpoints/lead2_cnsde",
        **kwargs,
    ):
        self.batch_size = batch_size
        mc = num_samples if num_samples is not None else (monte_carlo_samples if monte_carlo_samples is not None else 8)
        self.num_samples = mc
        self.monte_carlo_samples = mc
        self.learning_rate = learning_rate if learning_rate is not None else (lr if lr is not None else 0.001)
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip if grad_clip is not None else (clip_grad if clip_grad is not None else 1.0)
        self.mixed_precision = mixed_precision
        self.epochs = epochs
        self.seed = seed
        self.num_workers = num_workers
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.run_name = run_name
        self.checkpoint_dir = checkpoint_dir

    @property
    def lr(self) -> float:
        return self.learning_rate


@dataclass
class ValidationConfig:
    num_samples: int = 32
    monte_carlo_samples: int = 32

    def __init__(
        self,
        num_samples: Optional[int] = None,
        monte_carlo_samples: Optional[int] = None,
        **kwargs,
    ):
        val_mc = num_samples if num_samples is not None else (monte_carlo_samples if monte_carlo_samples is not None else 32)
        self.num_samples = val_mc
        self.monte_carlo_samples = val_mc


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    signature: SignatureConfig = field(default_factory=SignatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sde: SDEConfig = field(default_factory=SDEConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


def load_config(yaml_input: Any) -> Config:
    if isinstance(yaml_input, Config):
        return yaml_input

    if isinstance(yaml_input, dict):
        cfg_dict = yaml_input
        default_name = "lead2_cnsde"
    elif hasattr(yaml_input, "read"):
        cfg_dict = yaml.safe_load(yaml_input) or {}
        default_name = "lead2_cnsde"
    else:
        with open(yaml_input, "r") as f:
            cfg_dict = yaml.safe_load(f) or {}
        default_name = os.path.splitext(os.path.basename(yaml_input))[0]

    data_raw = cfg_dict.get("data", {})
    sig_raw = cfg_dict.get("signature", {})
    model_raw = cfg_dict.get("model", {})
    sde_raw = cfg_dict.get("sde", {})
    train_raw = cfg_dict.get("training", {})
    val_raw = cfg_dict.get("validation", {})

    data_cfg = DataConfig(**data_raw)
    sig_cfg = SignatureConfig(**sig_raw)
    
    # Propagate num_leads to model config if not explicitly set
    if "num_leads" not in model_raw:
        model_raw["num_leads"] = data_cfg.num_leads
    model_cfg = ModelConfig(**model_raw)
    
    sde_cfg = SDEConfig(**sde_raw)

    if "run_name" not in train_raw:
        train_raw["run_name"] = default_name
    train_cfg = TrainingConfig(**train_raw)
    val_cfg = ValidationConfig(**val_raw)

    return Config(
        data=data_cfg,
        signature=sig_cfg,
        model=model_cfg,
        sde=sde_cfg,
        training=train_cfg,
        validation=val_cfg,
    )


# Attempt to register PyTorch 2.6+ safe globals so unpickling Config works seamlessly
try:
    import torch
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([
            Config,
            DataConfig,
            SignatureConfig,
            ModelConfig,
            SDEConfig,
            TrainingConfig,
            ValidationConfig,
        ])
except Exception:
    pass
