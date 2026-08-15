"""Dataset layer — MaterialRecord + JSONL dataset helpers."""
from __future__ import annotations
import json
import time
import uuid
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from matgraph.client import fetch_materials_data

class MaterialRecord(BaseModel):
    """Dataset layer: single material entry with ML + DFT provenance.

    Combines structure, ML prediction (E_form, E_hull, uncertainty),
    DFT results (when available), and provenance.
    """
    material_id: str = Field(default_factory=lambda: f"al-{uuid.uuid4().hex[:8]}")
    formula: str
    structure: Optional[Any] = None  # pymatgen Structure or dict
    formation_energy_per_atom: Optional[float] = None
    energy_above_hull: Optional[float] = None
    band_gap: Optional[float] = None
    uncertainty: Optional[float] = None
    acquisition_score: Optional[float] = None
    acquisition: Optional[str] = None  # max_uncertainty | ei | ucb
    source: str = "ml"  # ml | dft | ai
    dft_code: Optional[str] = None  # vasp | qe
    dft_energy: Optional[float] = None
    dft_converged: Optional[bool] = None
    created_at: float = Field(default_factory=time.time)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        # Structure: serialize to dict if it's a real pymatgen Structure; mock structures -> None
        try:
            if self.structure is not None and hasattr(self.structure, "as_dict"):
                # Guard against MagicMock which returns mock for as_dict
                if type(self.structure).__name__ == "MagicMock":
                    d["structure"] = None
                else:
                    val = self.structure.as_dict()
                    # Ensure it's JSON serializable (must be dict)
                    if isinstance(val, dict):
                        d["structure"] = val
                    else:
                        d["structure"] = None
            else:
                # Non-structure types: try dict, else None
                if self.structure is not None and not isinstance(self.structure, dict):
                    # e.g., MagicMock ->None
                    try:
                        json.dumps(self.structure)
                    except Exception:
                        d["structure"] = None
        except Exception:
            d["structure"] = None
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MaterialRecord":
        # If structure is dict with lattice, try to restore as Structure
        sd = d.get("structure")
        if isinstance(sd, dict) and "lattice" in sd:
            try:
                from pymatgen.core import Structure
                d = dict(d)
                d["structure"] = Structure.from_dict(sd)
            except Exception:
                pass
        return cls(**d)

    def hash(self) -> str:
        """Deterministic hash for deduplication."""
        h = hashlib.sha256(f"{self.formula}:{self.formation_energy_per_atom}:{self.energy_above_hull}".encode()).hexdigest()[:12]
        return h

# Dataset helpers — JSONL file

def _dataset_path(path: str | Path) -> Path:
    return Path(path)

def save_dataset(records: List[MaterialRecord], path: str | Path) -> str:
    """Save list of MaterialRecords to JSONL."""
    p = _dataset_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in records:
            json.dump(r.to_dict(), f)
            f.write("\n")
    return str(p)

def load_dataset(path: str | Path) -> List[MaterialRecord]:
    """Load JSONL dataset."""
    p = _dataset_path(path)
    if not p.exists():
        return []
    out: List[MaterialRecord] = []
    with open(p) as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                d=json.loads(line)
                out.append(MaterialRecord.from_dict(d))
            except Exception:
                continue
    return out

def append_records(records: List[MaterialRecord], path: str | Path) -> str:
    """Append records to existing JSONL (create if missing)."""
    p = _dataset_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        for r in records:
            json.dump(r.to_dict(), f)
            f.write("\n")
    return str(p)

def records_from_predictions(preds: List[Dict[str, Any]], acquisition: str = "max_uncertainty") -> List[MaterialRecord]:
    """Convert run_pipeline predictions to MaterialRecords."""
    out=[]
    for p in preds:
        rec = MaterialRecord(
            material_id=str(p.get("material_id", f"pred-{uuid.uuid4().hex[:6]}")),
            formula=p.get("formula", "Unknown"),
            structure=p.get("structure"),
            formation_energy_per_atom=p.get("predicted_form_energy") if p.get("predicted_form_energy") is not None else p.get("true_form_energy"),
            energy_above_hull=p.get("energy_above_hull"),
            band_gap=p.get("predicted_band_gap") if p.get("predicted_band_gap") is not None else p.get("true_band_gap"),
            uncertainty=p.get("uncertainty") or p.get("ml_uncertainty"),
            acquisition=acquisition,
            source="ml",
            provenance=p.get("provenance", {}),
        )
        out.append(rec)
    return out

__all__ = ["MaterialRecord", "fetch_materials_data", "save_dataset", "load_dataset", "append_records", "records_from_predictions"]
