"""Training loop for crystal models: fit / evaluate / predict / checkpoint.

Every number reported comes from the data passed in. Nothing is simulated.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from matgraph.nn.models import CrystalGNN

_LOSSES = {"mse": F.mse_loss, "l1": F.l1_loss, "huber": F.smooth_l1_loss}


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    """Fit a ``CrystalGNN`` (or any module taking a ``CrystalBatch``) on a target."""

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        loss: str = "mse",
        device: Union[str, torch.device, None] = None,
    ):
        if loss not in _LOSSES:
            raise ValueError(f"loss must be one of {sorted(_LOSSES)}, got {loss!r}")
        self.device = torch.device(device) if device is not None else _default_device()
        self.model = model.to(self.device)
        self.loss_name = loss
        self.loss_fn = _LOSSES[loss]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.history: List[Dict[str, Any]] = []

    def _check_target(self, pred: torch.Tensor, target: Optional[torch.Tensor]) -> torch.Tensor:
        if target is None:
            raise ValueError("training batches need targets; build graphs with target=...")
        if target.shape != pred.shape:
            raise ValueError(f"target shape {tuple(target.shape)} does not match prediction {tuple(pred.shape)}")
        return target

    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        total_loss, total_abs, n_crystals, n_values = 0.0, 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for batch in loader:
                batch = batch.to(self.device)
                pred = self.model(batch)
                target = self._check_target(pred, batch.target)
                loss = self.loss_fn(pred, target)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item() * target.shape[0]
                total_abs += (pred - target).abs().sum().item()
                n_crystals += target.shape[0]
                n_values += target.numel()
        if n_crystals == 0:
            raise ValueError("loader produced no samples")
        return {"loss": total_loss / n_crystals, "mae": total_abs / n_values}

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Loss and mean absolute error on a loader, with no weight updates."""
        return self._run_epoch(loader, train=False)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        patience: Optional[int] = None,
        on_epoch_end: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Train for up to ``epochs``. With validation data and ``patience``, stop early
        and restore the weights from the best validation epoch."""
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        if patience is not None and val_loader is None:
            raise ValueError("patience requires a val_loader")
        best_val, best_state, best_epoch, stale = float("inf"), None, 0, 0

        for epoch in range(1, epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True)
            record: Dict[str, Any] = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}}
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                record.update({f"val_{k}": v for k, v in val_metrics.items()})
                if val_metrics["loss"] < best_val:
                    best_val, best_epoch, stale = val_metrics["loss"], epoch, 0
                    best_state = copy.deepcopy(self.model.state_dict())
                else:
                    stale += 1
            self.history.append(record)
            if on_epoch_end is not None:
                on_epoch_end(record)
            if patience is not None and stale >= patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            for record in self.history:
                record["best_epoch"] = best_epoch
        return self.history

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> torch.Tensor:
        """Predictions for every crystal in the loader, shape (num_crystals, num_targets)."""
        self.model.eval()
        outs = [self.model(batch.to(self.device)).cpu() for batch in loader]
        if not outs:
            raise ValueError("loader produced no samples")
        return torch.cat(outs, dim=0)

    def save(self, path: Union[str, Path]) -> Path:
        """Write config, weights and training history to one file."""
        if not isinstance(self.model, CrystalGNN):
            raise TypeError("save() supports CrystalGNN models only; save the state_dict yourself otherwise")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "matgraph.nn.v1",
                "model_config": self.model.config,
                "state_dict": self.model.state_dict(),
                "history": self.history,
                "loss": self.loss_name,
            },
            path,
        )
        return path


def load_checkpoint(path: Union[str, Path], device: Union[str, torch.device, None] = None) -> CrystalGNN:
    """Rebuild a ``CrystalGNN`` saved by ``Trainer.save``, with its training history attached."""
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
    if ckpt.get("format") != "matgraph.nn.v1":
        raise ValueError(f"{path} is not a matgraph.nn checkpoint")
    model = CrystalGNN(**ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.history = ckpt["history"]  # type: ignore[attr-defined]
    dev = torch.device(device) if device is not None else _default_device()
    return model.to(dev)
