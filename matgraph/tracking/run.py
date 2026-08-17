"""Run — wandb.init/log/finish compatible."""
from __future__ import annotations
from typing import Optional, Dict, Any
from .store import create_run, log_metrics, log_artifact, finish_run

_CURRENT = None

class Run:
    def __init__(self, project: str="matgraph", name: Optional[str]=None, config: Optional[Dict[str,Any]]=None):
        self.project = project
        self.name = name
        self.config = config or {}
        self.id = create_run(project, name, self.config)
        self._step = 0
        global _CURRENT
        _CURRENT = self
        # auto wandb sync — offline if no key, online if user set WANDB_API_KEY
        self._wb = None
        try:
            import os
            from matgraph.config import get_wandb_key, get_config_value
            key = get_wandb_key()
            base = get_config_value("wandb_base_url")
            if base:
                os.environ["WANDB_BASE_URL"] = base
            if key:
                os.environ["WANDB_API_KEY"] = key
                os.environ.pop("WANDB_MODE", None)
            else:
                os.environ.setdefault("WANDB_MODE", "offline")
                os.environ.setdefault("WANDB_DISABLE_CODE", "true")
            import wandb
            self._wb = wandb.init(project=project, name=name, config=self.config, reinit=True)
        except Exception:
            pass

    def log(self, metrics: Dict[str,Any], step: Optional[int]=None):
        if step is None:
            self._step += 1
            step = self._step
        log_metrics(self.id, metrics, step=step)
        if self._wb:
            try:
                self._wb.log(metrics, step=step)
            except Exception:
                pass
        # also print for local
        return metrics

    def log_artifact(self, path: str, type: str="dataset"):
        log_artifact(self.id, path, typ=type)
        if self._wb:
            try:
                import wandb
                art = wandb.Artifact(name=path.split("/")[-1], type=type)
                art.add_file(path)
                self._wb.log_artifact(art)
            except Exception:
                pass

    def log_table(self, name: str, columns: list, data: list):
        # wandb.Table compatible — store as json artifact + log
        import json, pathlib, tempfile
        tbl = {"columns": columns, "data": data}
        p = pathlib.Path(tempfile.gettempdir()) / f"matgraph_table_{self.id}_{name}.json"
        p.write_text(json.dumps(tbl))
        self.log_artifact(str(p), type="table")
        self.log({f"table/{name}": len(data)})
        # also try wandb.Table
        if self._wb:
            try:
                import wandb
                wt = wandb.Table(columns=columns, data=data)
                self._wb.log({f"table/{name}": wt})
            except Exception:
                pass

    def log_image(self, name: str, path: str):
        self.log_artifact(path, type="image")
        self.log({f"image/{name}": path})
        if self._wb:
            try:
                import wandb
                self._wb.log({f"image/{name}": wandb.Image(path)})
            except Exception:
                pass

    # wandb-compatible: wandb.Table / wandb.Artifact / wandb.Image passthrough
    def __getattr__(self, name):
        if self._wb and hasattr(self._wb, name):
            return getattr(self._wb, name)
        raise AttributeError(name)

    def finish(self):
        finish_run(self.id)
        if self._wb:
            try:
                self._wb.finish()
            except Exception:
                pass
        global _CURRENT
        if _CURRENT is self:
            _CURRENT = None

def init(project: str="matgraph", name: Optional[str]=None, config: Optional[Dict[str,Any]]=None) -> Run:
    return Run(project=project, name=name, config=config)

def current_run() -> Optional[Run]:
    return _CURRENT
