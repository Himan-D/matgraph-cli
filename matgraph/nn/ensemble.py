"""Deep ensembles: uncertainty as the spread of independently trained models."""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from matgraph.nn.models import CrystalGNN
from matgraph.nn.train import Trainer


def train_ensemble(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    n_members: int,
    epochs: int,
    make_model: Callable[[], CrystalGNN] = CrystalGNN,
    lr: float = 1e-3,
    patience: Optional[int] = None,
    seed: int = 0,
    device: Optional[str] = None,
) -> List[CrystalGNN]:
    """Train ``n_members`` models from different seeds on the same data."""
    if n_members < 2:
        raise ValueError("an ensemble needs at least 2 members to estimate spread")
    members = []
    for k in range(n_members):
        torch.manual_seed(seed + k)
        trainer = Trainer(make_model(), lr=lr, device=device)
        trainer.fit(train_loader, val_loader=val_loader, epochs=epochs, patience=patience)
        members.append(trainer.model.cpu())
    return members


def predict_ensemble(
    models: List[torch.nn.Module],
    loader: DataLoader,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean and standard deviation over members, each shape (num_crystals, num_targets)."""
    if len(models) < 2:
        raise ValueError("predict_ensemble needs at least 2 models; a single model has no spread")
    preds = torch.stack([Trainer(m, device=device).predict(loader) for m in models])
    return preds.mean(dim=0), preds.std(dim=0, unbiased=False)
