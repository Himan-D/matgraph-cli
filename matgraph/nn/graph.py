"""Crystal graph construction and batching — periodic neighbor search in numpy.

No pymatgen or DGL is needed: a structure is described by its lattice matrix,
fractional coordinates and atomic numbers (a plain dict, see ``structure_to_dict``).
``from_pymatgen`` adapts a pymatgen ``Structure`` by duck typing when pymatgen is installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class CrystalGraph:
    """One crystal as a directed neighbor graph.

    edge_index[0] is the neighbor (source) atom, edge_index[1] the center
    (target) atom that receives the message. Periodic images are separate edges.
    """

    atomic_numbers: torch.Tensor  # (N,) long, values in [1, 118]
    edge_index: torch.Tensor  # (2, E) long
    edge_distance: torch.Tensor  # (E,) float
    target: Optional[torch.Tensor] = None  # (T,) float


@dataclass
class CrystalBatch:
    """Several crystals concatenated into one graph, with per-atom crystal ids."""

    atomic_numbers: torch.Tensor  # (N,)
    edge_index: torch.Tensor  # (2, E)
    edge_distance: torch.Tensor  # (E,)
    crystal_index: torch.Tensor  # (N,) which crystal each atom belongs to
    num_crystals: int
    target: Optional[torch.Tensor] = None  # (B, T)

    def to(self, device: torch.device | str) -> "CrystalBatch":
        return CrystalBatch(
            atomic_numbers=self.atomic_numbers.to(device),
            edge_index=self.edge_index.to(device),
            edge_distance=self.edge_distance.to(device),
            crystal_index=self.crystal_index.to(device),
            num_crystals=self.num_crystals,
            target=None if self.target is None else self.target.to(device),
        )


def build_crystal_graph(
    lattice: Sequence[Sequence[float]],
    frac_coords: Sequence[Sequence[float]],
    atomic_numbers: Sequence[int],
    cutoff: float = 6.0,
    max_neighbors: int = 12,
    target: Optional[Sequence[float]] = None,
) -> CrystalGraph:
    """Build the periodic neighbor graph of one crystal.

    Every atom keeps at most ``max_neighbors`` neighbors within ``cutoff``,
    nearest first. Periodic images are searched in all directions far enough
    to cover the cutoff, so non-orthogonal and small cells are handled.
    Memory is O(N * S) per atom (S = number of images searched), not O(N^2 * S).
    """
    lat = np.asarray(lattice, dtype=float)
    frac = np.asarray(frac_coords, dtype=float)
    z = np.asarray(atomic_numbers, dtype=np.int64)

    if lat.shape != (3, 3):
        raise ValueError(f"lattice must be 3x3, got shape {lat.shape}")
    if frac.ndim != 2 or frac.shape[1] != 3:
        raise ValueError(f"frac_coords must be (N, 3), got shape {frac.shape}")
    if z.ndim != 1 or len(z) != len(frac):
        raise ValueError("atomic_numbers must be 1-D with one entry per site")
    if len(z) == 0:
        raise ValueError("structure has no sites")
    if np.any(z < 1) or np.any(z > 118):
        raise ValueError("atomic numbers must be in [1, 118]")
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    if max_neighbors < 1:
        raise ValueError("max_neighbors must be >= 1")

    n = len(z)
    # Wrap into the unit cell so the image search below covers every neighbor.
    cart = (frac % 1.0) @ lat

    # Plane spacing along each lattice direction decides how many images to try.
    recip = np.linalg.inv(lat).T
    spacing = 1.0 / np.linalg.norm(recip, axis=1)
    reps = np.ceil(cutoff / spacing).astype(int)
    grids = np.meshgrid(*[np.arange(-r, r + 1) for r in reps], indexing="ij")
    shifts = np.stack(grids, axis=-1).reshape(-1, 3) @ lat  # (S, 3) cartesian

    centers, neighbors, distances = [], [], []
    for i in range(n):
        # diff[s, j] = position of image s of atom j relative to atom i
        diff = cart[None, :, :] + shifts[:, None, :] - cart[i]
        dist = np.linalg.norm(diff, axis=-1).reshape(-1)  # flat index = s * n + j
        valid = np.flatnonzero((dist > 1e-8) & (dist <= cutoff))
        valid = valid[np.argsort(dist[valid], kind="stable")][:max_neighbors]
        centers.append(np.full(len(valid), i, dtype=np.int64))
        neighbors.append(valid % n)
        distances.append(dist[valid])

    center = np.concatenate(centers)
    neighbor = np.concatenate(neighbors)
    d = np.concatenate(distances)

    edge_index = torch.tensor(np.stack([neighbor, center]), dtype=torch.long).reshape(2, -1)
    return CrystalGraph(
        atomic_numbers=torch.tensor(z, dtype=torch.long),
        edge_index=edge_index,
        edge_distance=torch.tensor(d, dtype=torch.float32),
        target=None if target is None else torch.tensor(np.atleast_1d(target), dtype=torch.float32),
    )


def structure_to_dict(structure) -> Dict[str, Any]:
    """Serialize a pymatgen ``Structure`` (duck-typed) to the plain-dict format."""
    return {
        "lattice": np.asarray(structure.lattice.matrix).tolist(),
        "frac_coords": np.asarray(structure.frac_coords).tolist(),
        "atomic_numbers": [int(site.specie.Z) for site in structure],
    }


def graph_from_dict(
    structure: Dict[str, Any],
    cutoff: float = 6.0,
    max_neighbors: int = 12,
    target: Optional[Sequence[float]] = None,
) -> CrystalGraph:
    """Build a graph from ``{"lattice", "frac_coords", "atomic_numbers"}``."""
    missing = {"lattice", "frac_coords", "atomic_numbers"} - set(structure)
    if missing:
        raise ValueError(f"structure dict is missing keys: {sorted(missing)}")
    return build_crystal_graph(
        lattice=structure["lattice"],
        frac_coords=structure["frac_coords"],
        atomic_numbers=structure["atomic_numbers"],
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        target=target,
    )


def from_pymatgen(structure, cutoff: float = 6.0, max_neighbors: int = 12, target=None) -> CrystalGraph:
    """Adapter for a pymatgen ``Structure`` (imported lazily; duck-typed)."""
    return graph_from_dict(structure_to_dict(structure), cutoff=cutoff, max_neighbors=max_neighbors, target=target)


def collate_crystals(graphs: Sequence[CrystalGraph]) -> CrystalBatch:
    """Merge crystals into one batch; edge indices are shifted per crystal."""
    if len(graphs) == 0:
        raise ValueError("cannot collate an empty list of graphs")
    has_target = [g.target is not None for g in graphs]
    if any(has_target) and not all(has_target):
        raise ValueError("either every graph in a batch has a target or none does")

    zs, edges, dists, crystal_ids, targets = [], [], [], [], []
    offset = 0
    for k, g in enumerate(graphs):
        n = g.atomic_numbers.shape[0]
        zs.append(g.atomic_numbers)
        edges.append(g.edge_index + offset)
        dists.append(g.edge_distance)
        crystal_ids.append(torch.full((n,), k, dtype=torch.long))
        if g.target is not None:
            targets.append(g.target)
        offset += n

    return CrystalBatch(
        atomic_numbers=torch.cat(zs),
        edge_index=torch.cat(edges, dim=1),
        edge_distance=torch.cat(dists),
        crystal_index=torch.cat(crystal_ids),
        num_crystals=len(graphs),
        target=torch.stack(targets) if targets else None,
    )


class CrystalDataset(Dataset):
    """In-memory list of ``CrystalGraph`` objects.

    Use with ``DataLoader(ds, batch_size=32, collate_fn=collate_crystals)``.
    """

    def __init__(self, graphs: Sequence[CrystalGraph]):
        self.graphs = list(graphs)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> CrystalGraph:
        return self.graphs[idx]
