"""Message-passing layers for crystal graphs (CGCNN-style gated convolution)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianExpansion(nn.Module):
    """Expand scalar bond lengths onto a fixed grid of Gaussian basis functions."""

    def __init__(self, dmin: float = 0.0, dmax: float = 6.0, num_gaussians: int = 64):
        super().__init__()
        if num_gaussians < 2:
            raise ValueError("num_gaussians must be >= 2")
        if dmax <= dmin:
            raise ValueError("dmax must be greater than dmin")
        self.register_buffer("centers", torch.linspace(dmin, dmax, num_gaussians))
        self.width = float((dmax - dmin) / (num_gaussians - 1))

    @property
    def out_dim(self) -> int:
        return int(self.centers.numel())

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        diff = distance.reshape(-1, 1) - self.centers.reshape(1, -1)
        return torch.exp(-(diff**2) / self.width**2)


class CGConv(nn.Module):
    """Crystal graph convolution (Xie & Grossman, PRL 2018).

    For each edge j->i: z = [x_i, x_j, e_ij], message = sigmoid(filter) * softplus(core).
    Messages are summed per center atom and added residually.

    LayerNorm replaces CGCNN's BatchNorm so a single-atom batch still works and
    eval-mode predictions do not depend on batch statistics.
    """

    def __init__(self, atom_dim: int, edge_dim: int):
        super().__init__()
        self.atom_dim = atom_dim
        self.fc = nn.Linear(2 * atom_dim + edge_dim, 2 * atom_dim)
        self.norm = nn.LayerNorm(atom_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        z = torch.cat([x[dst], x[src], edge_attr], dim=1)
        filt, core = self.fc(z).chunk(2, dim=1)
        msg = torch.sigmoid(filt) * F.softplus(core)
        agg = x.new_zeros(x.shape).index_add(0, dst, msg)
        return F.softplus(x + self.norm(agg))
