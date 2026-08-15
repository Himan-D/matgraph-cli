"""Thin, testable Materials Project client — no ML logic here."""
from __future__ import annotations
from typing import Optional, Tuple, List
from matgraph.exceptions import DataNotFoundError

def fetch_materials_data(
    formula: str,
    api_key: str,
    band_gap_range: Optional[Tuple[float, float]] = None,
    crystal_system: Optional[str] = None,
):
    search_kwargs = {
        "formula": formula,
        "fields": ["material_id", "formula_pretty", "structure", "band_gap", "formation_energy_per_atom", "density", "symmetry", "energy_above_hull", "is_stable"],
    }
    if band_gap_range:
        search_kwargs["band_gap"] = band_gap_range
    if crystal_system:
        search_kwargs["crystal_system"] = crystal_system
    from mp_api.client import MPRester
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(**search_kwargs)
    return docs
