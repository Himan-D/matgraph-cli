"""W&B experiment tracking for materials."""
from .run import Run, init, current_run
from .store import list_runs, get_run

try:
    import wandb
    _HAS_WANDB = True
    Artifact = wandb.Artifact
    Table = wandb.Table
    Image = wandb.Image
    config = wandb.config
    log = wandb.log
    log_artifact = wandb.log_artifact
    finish = wandb.finish
except ImportError:
    _HAS_WANDB = False
    Artifact = Table = Image = config = log = log_artifact = finish = None

__all__ = ["Run","init","current_run","list_runs","get_run","Artifact","Table","Image","config","log","log_artifact","finish","_HAS_WANDB"]
