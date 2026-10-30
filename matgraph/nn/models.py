"""Crystal property models built from ``nn.Module`` layers."""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from matgraph.nn.graph import CrystalBatch
from matgraph.nn.layers import CGConv, GaussianExpansion


class CrystalGNN(nn.Module):
    """Crystal graph network: embedding -> CGConv stack -> mean pool -> MLP head.

    Output shape is (num_crystals, num_targets). Use ``config`` to rebuild the
    same architecture (checkpoints store it).
    """

    def __init__(
        self,
        atom_dim: int = 64,
        num_conv: int = 3,
        cutoff: float = 6.0,
        num_gaussians: int = 64,
        hidden_dim: int = 128,
        num_targets: int = 1,
        max_atomic_number: int = 118,
    ):
        super().__init__()
        if num_conv < 1:
            raise ValueError("num_conv must be >= 1")
        if num_targets < 1:
            raise ValueError("num_targets must be >= 1")
        self.config: Dict[str, Any] = dict(
            atom_dim=atom_dim,
            num_conv=num_conv,
            cutoff=cutoff,
            num_gaussians=num_gaussians,
            hidden_dim=hidden_dim,
            num_targets=num_targets,
            max_atomic_number=max_atomic_number,
        )
        self.max_atomic_number = max_atomic_number
        self.embedding = nn.Embedding(max_atomic_number + 1, atom_dim, padding_idx=0)
        self.expand = GaussianExpansion(0.0, cutoff, num_gaussians)
        self.convs = nn.ModuleList(CGConv(atom_dim, num_gaussians) for _ in range(num_conv))
        self.head = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_targets),
        )

    def forward(self, batch: CrystalBatch) -> torch.Tensor:
        z = batch.atomic_numbers
        if z.numel() and (z.min() < 1 or z.max() > self.max_atomic_number):
            raise ValueError(f"atomic numbers must be in [1, {self.max_atomic_number}]")
        x = self.embedding(z)
        edge_attr = self.expand(batch.edge_distance)
        for conv in self.convs:
            x = conv(x, batch.edge_index, edge_attr)

        # Mean-pool atoms into one vector per crystal.
        pooled = x.new_zeros(batch.num_crystals, x.shape[1]).index_add(0, batch.crystal_index, x)
        counts = torch.bincount(batch.crystal_index, minlength=batch.num_crystals)
        pooled = pooled / counts.clamp(min=1).unsqueeze(1).to(x.dtype)
        return self.head(pooled)
