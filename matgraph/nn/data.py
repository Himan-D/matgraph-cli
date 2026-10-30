"""Load crystal datasets from the repo's formats: CSV (JSON or CIF structures) and MaterialRecord."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from matgraph.exceptions import ConfigError, ValidationError
from matgraph.nn.graph import CrystalGraph, CrystalDataset, graph_from_dict, structure_to_dict


def _read_structure_file(path: Path) -> dict:
    try:
        from pymatgen.core import Structure
    except ImportError as e:
        raise ConfigError("CIF input needs pymatgen — pip install pymatgen, or use a 'structure' JSON column") from e
    return structure_to_dict(Structure.from_file(str(path)))


def load_crystal_dataset(
    csv_path: str | Path,
    target: str,
    cutoff: float = 6.0,
    max_neighbors: int = 12,
) -> CrystalDataset:
    """Read a CSV where each row is one crystal and ``target`` is a numeric column.

    The structure comes from either:
      * ``structure``: a JSON object ``{"lattice", "frac_coords", "atomic_numbers"}``, or
      * ``cif``: a path to a CIF file, relative to the CSV's folder (needs pymatgen).

    Rows with a missing or non-numeric target raise instead of being dropped.
    """
    path = Path(csv_path)
    if not path.is_file() or path.suffix.lower() != ".csv":
        raise ValidationError(f"expected a CSV file, got {csv_path}")
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        if target not in cols:
            raise ValidationError(f"target column {target!r} not in {path.name} (columns: {cols})")
        if "structure" not in cols and "cif" not in cols:
            raise ValidationError(f"{path.name} needs a 'structure' (JSON) or 'cif' (path) column")
        rows = list(reader)
    if not rows:
        raise ValidationError(f"{path.name} has no rows")

    graphs: List[CrystalGraph] = []
    for line_no, row in enumerate(rows, start=2):
        try:
            y = float(row[target])
        except (TypeError, ValueError):
            raise ValidationError(f"{path.name}:{line_no}: target {target!r} is not a number: {row[target]!r}") from None
        if row.get("structure"):
            try:
                structure = json.loads(row["structure"])
            except json.JSONDecodeError as e:
                raise ValidationError(f"{path.name}:{line_no}: invalid structure JSON: {e}") from e
        elif row.get("cif"):
            structure = _read_structure_file(path.parent / row["cif"])
        else:
            raise ValidationError(f"{path.name}:{line_no}: no structure or cif given")
        graphs.append(graph_from_dict(structure, cutoff=cutoff, max_neighbors=max_neighbors, target=[y]))
    return CrystalDataset(graphs)


def _normalize_structure(structure: Any) -> dict:
    """Accept the plain format, or pymatgen's ``Structure.as_dict()`` (used by MaterialRecord)."""
    if not isinstance(structure, dict):
        raise ValidationError("structure must be a JSON object")
    if "lattice" in structure and "frac_coords" in structure:
        return structure
    if "sites" in structure and isinstance(structure.get("lattice"), dict):
        try:
            from pymatgen.core import Element
        except ImportError as e:
            raise ConfigError("pymatgen-format structures need pymatgen — pip install pymatgen") from e
        sites = structure["sites"]
        return {
            "lattice": structure["lattice"]["matrix"],
            "frac_coords": [s["abc"] for s in sites],
            "atomic_numbers": [Element(s["species"][0]["element"]).Z for s in sites],
        }
    raise ValidationError("structure needs 'lattice'+'frac_coords'+'atomic_numbers' or pymatgen 'sites' format")


def load_jsonl_dataset(
    jsonl_path: str | Path,
    target: str,
    cutoff: float = 6.0,
    max_neighbors: int = 12,
) -> CrystalDataset:
    """Read a JSONL file of MaterialRecord dicts (one per line), e.g. the active-learning dataset."""
    path = Path(jsonl_path)
    graphs: List[CrystalGraph] = []
    with path.open() as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValidationError(f"{path.name}:{line_no}: invalid JSON: {e}") from e
            if rec.get("structure") is None:
                raise ValidationError(f"{path.name}:{line_no}: record has no structure")
            if rec.get(target) is None:
                raise ValidationError(f"{path.name}:{line_no}: record has no {target!r} value")
            structure = _normalize_structure(rec["structure"])
            graphs.append(graph_from_dict(structure, cutoff=cutoff, max_neighbors=max_neighbors, target=[float(rec[target])]))
    if not graphs:
        raise ValidationError(f"{path.name} has no records")
    return CrystalDataset(graphs)


def load_dataset(path: str | Path, target: str, cutoff: float = 6.0, max_neighbors: int = 12) -> CrystalDataset:
    """Dispatch on file type: ``.csv`` or ``.jsonl``."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return load_crystal_dataset(p, target, cutoff=cutoff, max_neighbors=max_neighbors)
    if p.suffix.lower() == ".jsonl":
        return load_jsonl_dataset(p, target, cutoff=cutoff, max_neighbors=max_neighbors)
    raise ValidationError(f"unsupported dataset type {p.suffix!r}: use .csv or .jsonl")


def graphs_from_records(
    records: Iterable[Any],
    target: str,
    cutoff: float = 6.0,
    max_neighbors: int = 12,
) -> CrystalDataset:
    """Build a dataset from ``MaterialRecord`` objects.

    ``record.structure`` may be a dict in the plain format or a pymatgen Structure;
    ``target`` names a numeric field on the record (e.g. ``formation_energy_per_atom``).
    Records without a structure or target raise.
    """
    graphs: List[CrystalGraph] = []
    for rec in records:
        structure = getattr(rec, "structure", None)
        y = getattr(rec, target, None)
        if structure is None:
            raise ValidationError(f"record {getattr(rec, 'material_id', '?')} has no structure")
        if y is None:
            raise ValidationError(f"record {getattr(rec, 'material_id', '?')} has no {target!r} value")
        if not isinstance(structure, dict):
            structure = structure_to_dict(structure)
        graphs.append(graph_from_dict(structure, cutoff=cutoff, max_neighbors=max_neighbors, target=[float(y)]))
    if not graphs:
        raise ValidationError("no records given")
    return CrystalDataset(graphs)


def train_val_split(
    graphs: Sequence[CrystalGraph],
    val_fraction: float = 0.2,
    seed: int = 0,
) -> Tuple[List[CrystalGraph], List[CrystalGraph]]:
    """Seeded shuffle split. Needs at least 5 crystals so both sides are non-trivial."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    if len(graphs) < 5:
        raise ValueError(f"need at least 5 crystals to split into train/val, got {len(graphs)}")
    idx = list(range(len(graphs)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(round(len(graphs) * val_fraction)))
    val_idx, train_idx = set(idx[:n_val]), idx[n_val:]
    train = [graphs[i] for i in train_idx]
    val = [graphs[i] for i in sorted(val_idx)]
    return train, val
