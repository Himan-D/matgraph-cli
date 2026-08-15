"""Drop-in wandb compatibility — `import matgraph.tracking as wandb` works."""
from __future__ import annotations
from typing import Optional, Dict, Any, List
import os

# Re-export wandb-like API with local fallback
try:
    import wandb as _real_wandb
    _HAS_WANDB = True
except Exception:
    _real_wandb = None  # type: ignore
    _HAS_WANDB = False

from .run import Run, init, current_run
from .store import list_runs, get_run

# wandb.config-like global
class _Config(dict):
    def __init__(self, d: Dict[str,Any]):
        super().__init__(d)
    def update(self, d: Dict[str,Any]):  # type: ignore
        super().update(d)

config = _Config({})

# wandb.Artifact compatible
class Artifact:
    def __init__(self, name: str, type: str = "dataset", description: str = ""):
        self.name = name
        self.type = type
        self.description = description
        self._files: List[str] = []
    def add_file(self, path: str):
        self._files.append(path)
    def add_dir(self, path: str):
        import pathlib
        for p in pathlib.Path(path).rglob("*"):
            if p.is_file():
                self._files.append(str(p))

# wandb.Table compatible
class Table:
    def __init__(self, columns: List[str]=None, data: List[List[Any]]=None):
        self.columns = columns or []
        self.data = data or []
    def add_data(self, *row):
        self.data.append(list(row))
    def to_json(self):
        import json
        return json.dumps({"columns": self.columns, "data": self.data})

# wandb.Image compatible (just path wrapper)
class Image:
    def __init__(self, path: str, caption: str=""):
        self.path = path
        self.caption = caption

# Top-level wandb.* functions that delegate to real wandb if installed, else local
def _get_run():
    return current_run()

def log(data: Dict[str,Any], step: Optional[int]=None):
    r = _get_run()
    if r:
        return r.log(data, step=step)
    # no run — create ephemeral
    if _HAS_WANDB:
        try:
            _real_wandb.log(data, step=step)
            return
        except Exception:
            pass
    # fallback: print
    return data

def log_artifact(artifact: Artifact, aliases: List[str]=None):
    r = _get_run()
    if r and isinstance(artifact, Artifact):
        for f in artifact._files:
            r.log_artifact(f, type=artifact.type)
        return
    if _HAS_WANDB:
        try:
            _real_wandb.log_artifact(artifact, aliases=aliases or [])
            return
        except Exception:
            pass

def finish():
    r = _get_run()
    if r:
        r.finish()
    if _HAS_WANDB:
        try:
            _real_wandb.finish()
        except Exception:
            pass

# For `import matgraph.tracking as wandb` to work: expose wandb namespace
__all__ = ["init","Run","config","Artifact","Table","Image","log","log_artifact","finish","current_run","list_runs","get_run","_HAS_WANDB"]
