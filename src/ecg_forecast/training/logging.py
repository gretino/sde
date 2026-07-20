from typing import Dict, Any, Optional

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class Logger:
    """Handles metrics logging to console and W&B matching Section 14."""

    def __init__(
        self,
        use_wandb: bool = False,
        project_name: str = "ecg-natural-dynamics",
        run_name: Optional[str] = None,
        config: Optional[Any] = None,
    ):
        self.use_wandb = use_wandb and HAS_WANDB
        if self.use_wandb:
            wandb.init(project=project_name, name=run_name, config=config)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if self.use_wandb:
            wandb.log(metrics, step=step)

    def log_image(self, tag: str, image_path: str, step: Optional[int] = None):
        if self.use_wandb:
            wandb.log({tag: wandb.Image(image_path)}, step=step)

    def log_summary(self, stage: str, epoch: int, metrics: Dict[str, float]):
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"[Stage {stage} | Epoch {epoch:02d}] {metric_str}")

    def finish(self):
        if self.use_wandb:
            wandb.finish()
