"""Train or fine-tune a matgraph crystal model on your own data.

Real training only: metrics come from the held-out validation split, and the
registry entry points at the checkpoint that produced them.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from matgraph.exceptions import ModelLoadError, ValidationError
from matgraph.nn.data import load_dataset, train_val_split
from matgraph.nn.graph import CrystalDataset, collate_crystals
from matgraph.nn.models import CrystalGNN
from matgraph.nn.train import Trainer, load_checkpoint

# MatGL foundation models use a different architecture; their weights cannot load into CrystalGNN.
_MATGL_BASES = {"chgnet", "m3gnet", "megnet"}


def _make_base(base: str) -> CrystalGNN:
    if base == "scratch":
        return CrystalGNN()
    if base.lower() in _MATGL_BASES:
        raise ModelLoadError(
            f"base {base!r} is a MatGL foundation model with a different architecture; "
            "it cannot be fine-tuned by matgraph.nn. Use base='scratch' or a matgraph checkpoint (.pt)."
        )
    p = Path(base)
    if p.suffix == ".pt" and p.is_file():
        return load_checkpoint(p, device="cpu")
    raise ValidationError(f"unknown base {base!r}: use 'scratch' or path to a matgraph .pt checkpoint")


def finetune(
    data_path: str,
    base: str = "scratch",
    epochs: int = 20,
    project: str = "finetune",
    target: str = "formation_energy_per_atom",
    cutoff: float = 6.0,
    max_neighbors: int = 12,
    lr: float = 1e-3,
    batch_size: int = 32,
    patience: Optional[int] = 5,
    val_fraction: float = 0.2,
    seed: int = 0,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Train on a .csv or .jsonl dataset and register the best-validation checkpoint."""
    from matgraph.settings import settings
    from matgraph.tracking import init as track_init
    from matgraph.training.registry import register_model

    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    model = _make_base(base)
    if model.config["num_targets"] != 1:
        raise ValidationError("finetune trains a single target; base model must have num_targets=1")
    dataset = load_dataset(data_path, target, cutoff=cutoff, max_neighbors=max_neighbors)
    train_graphs, val_graphs = train_val_split(dataset.graphs, val_fraction=val_fraction, seed=seed)

    torch.manual_seed(seed)
    train_loader = DataLoader(
        CrystalDataset(train_graphs), batch_size=min(batch_size, len(train_graphs)),
        shuffle=True, collate_fn=collate_crystals,
    )
    val_loader = DataLoader(CrystalDataset(val_graphs), batch_size=batch_size, collate_fn=collate_crystals)

    trainer = Trainer(model, lr=lr)
    run = track_init(project=project, name=f"{target}-{base}", config={
        "data": str(data_path), "base": base, "epochs": epochs, "target": target,
        "lr": lr, "batch_size": batch_size, "seed": seed, "n_train": len(train_graphs), "n_val": len(val_graphs),
    })
    try:
        history = trainer.fit(
            train_loader, val_loader, epochs=epochs, patience=patience,
            on_epoch_end=lambda rec: run.log({k: v for k, v in rec.items() if k != "epoch"}, step=rec["epoch"]),
        )
    finally:
        run.finish()

    best_epoch = history[0]["best_epoch"]
    best = next(r for r in history if r["epoch"] == best_epoch)
    metrics = {
        "val_mae": best["val_mae"],
        "val_loss": best["val_loss"],
        "train_mae": best["train_mae"],
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "n_train": len(train_graphs),
        "n_val": len(val_graphs),
    }

    out_path = Path(out_dir) if out_dir else settings.cache_dir / "models"
    checkpoint = trainer.save(out_path / f"{project}-{int(time.time())}.pt")
    model_id = register_model(
        name=f"{project}-{target}", base=base, dataset=str(data_path),
        metrics=metrics, artifact_path=str(checkpoint),
    )
    return {
        "model_id": model_id,
        "checkpoint": str(checkpoint),
        "artifact": str(checkpoint),
        "metrics": metrics,
        "n": len(dataset.graphs),
        "history": history,
    }
