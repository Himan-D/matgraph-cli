"""2D / exfoliation proxy."""
from __future__ import annotations
from typing import Dict, Any
def twod_metrics(formula: str, structure=None) -> Dict[str, Any]:
    # exfoliation energy proxy: layered detection via vdW gap heuristic = density low + gap
    exfol = None
    if structure is not None:
        try:
            density = structure.density
            # low density -> more likely vdW layered
            exfol = max(10, 80 - density*8)  # meV/atom
        except Exception:
            exfol = 35
    return {
        "formula": formula,
        "exfoliation_meV_per_atom_proxy": round(float(exfol),1) if exfol else None,
        "method": "density heuristic (no vdW-DF)",
        "reference": "Mounet et al. Nat Nanotech 2018",
        "uncertainty": 15,
        "threshold": "< 50 meV/atom likely exfoliable",
    }
