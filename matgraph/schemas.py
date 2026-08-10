"""Pydantic v2 schemas — single source of truth for validation."""
from __future__ import annotations
from typing import Optional, List, Literal, Any
from pydantic import BaseModel, Field, field_validator
import re

VALID_CRYSTAL_SYSTEMS = {"triclinic","monoclinic","orthorhombic","tetragonal","trigonal","hexagonal","cubic"}

FORMULA_RE = re.compile(r"^([A-Z][a-z]?\d*)+$")

class PredictRequest(BaseModel):
    formula: str = Field(..., description="Chemical formula, e.g. LiFePO4")
    min_gap: Optional[float] = Field(None, description="eV")
    max_gap: Optional[float] = Field(None, description="eV")
    crystal_system: Optional[str] = None
    model: str = Field("m3gnet", description="m3gnet | chgnet (future) | alignn (future)")
    seed: Optional[int] = None

    @field_validator("min_gap", "max_gap")
    @classmethod
    def _gap_bounds(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        from matgraph.settings import settings
        if not (settings.schema_min_gap <= v <= settings.schema_max_gap):
            raise ValueError(f"gap must be in [{settings.schema_min_gap}, {settings.schema_max_gap}]")
        return v

    @field_validator("formula")
    @classmethod
    def _formula(cls, v: str) -> str:
        v = v.strip()
        if not FORMULA_RE.match(v):
            raise ValueError(f"Invalid formula syntax: {v}")
        return v

    @field_validator("crystal_system")
    @classmethod
    def _crystal(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        low = v.lower()
        if low not in VALID_CRYSTAL_SYSTEMS:
            raise ValueError(f"crystal_system must be one of {sorted(VALID_CRYSTAL_SYSTEMS)}")
        # Return canonical Capitalized form
        return low.capitalize()

    @field_validator("model")
    @classmethod
    def _model(cls, v: str) -> str:
        low = v.lower()
        if low not in {"m3gnet","chgnet","alignn","cgcnn","megnet"}:
            raise ValueError("model must be m3gnet | chgnet | alignn (cgcnn/megnet are legacy aliases for m3gnet)")
        # normalize legacy
        if low in {"cgcnn","megnet"}:
            return "m3gnet"
        return low

class MaterialFeaturesSchema(BaseModel):
    num_elements: int
    mean_atomic_mass: float
    volume: float
    density: float

class ProvenanceSchema(BaseModel):
    mp_api_version: Optional[str] = None
    matgl_version: Optional[str] = None
    m3gnet_pes_model: str = Field(default_factory=lambda: __import__("matgraph.settings", fromlist=["settings"]).settings.pes_model)
    m3gnet_eform_model: str = Field(default_factory=lambda: __import__("matgraph.settings", fromlist=["settings"]).settings.eform_model)
    timestamp_utc: str
    git_sha: Optional[str] = None
    device: str = "cpu"
    seed: Optional[int] = None
    band_gap_source: str = "mp_experimental"  # or "ml_model" when real model exists

class MaterialPredictionSchema(BaseModel):
    material_id: str
    formula: str
    crystal_system: str
    true_band_gap: Optional[float] = None
    predicted_band_gap: Optional[float] = None  # None until real ML band-gap model ships
    band_gap_source: str = "mp_experimental"
    band_gap_note: Optional[str] = None
    true_form_energy: Optional[float] = None
    predicted_form_energy: Optional[float] = None
    m3gnet_energy: Optional[float] = None
    m3gnet_forces: Optional[List[List[float]]] = None
    m3gnet_stresses: Optional[List[float]] = None
    features: MaterialFeaturesSchema
    model_used: str
    provenance: ProvenanceSchema

class StabilitySchema(BaseModel):
    material_id: str
    formula: str
    formation_energy_per_atom: Optional[float]
    energy_above_hull: float
    is_stable: Optional[bool]
    stability_label: Literal["Stable","Metastable","Unstable"]

class SubstituteResultSchema(BaseModel):
    original: dict
    hypothetical: dict
    is_more_stable: bool
    provenance: ProvenanceSchema

# For batch
class BatchPredictRequest(BaseModel):
    formulas: List[str] = Field(..., min_length=1, max_length=50)
    model: str = "m3gnet"
    crystal_system: Optional[str] = None
    seed: Optional[int] = None
