"""W&B-like experiment tracking for materials — local-first, wandb-compatible."""
# Drop-in: `import matgraph.tracking as wandb` or `import wandb` (if installed, syncs)
from .run import Run, init, current_run
from .store import list_runs, get_run
from .wandb_compat import Artifact, Table, Image, config, log as wandb_log, log_artifact as wandb_log_artifact, finish as wandb_finish, _HAS_WANDB
# expose wandb-like top-level for `import matgraph.tracking as wandb`
log = wandb_log
log_artifact = wandb_log_artifact
finish = wandb_finish
__all__ = ["Run","init","current_run","list_runs","get_run","Artifact","Table","Image","config","log","log_artifact","finish","_HAS_WANDB"]
