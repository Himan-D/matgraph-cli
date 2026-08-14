"""Defects — vacancy formation proxy via cohesive energy."""
from __future__ import annotations
from typing import Dict, Any
def defect_metrics(formula: str, formation_energy_per_atom: float | None = None) -> Dict[str, Any]:
    # vacancy formation ~ -formation_energy + bond strength heuristic
    e_vac = None
    if formation_energy_per_atom is not None:
        e_vac = -formation_energy_per_atom * 3 + 1.0  # eV
        e_vac = max(0.2, min(5, e_vac))
    return {
        "formula": formula,
        "vacancy_formation_eV_proxy": round(float(e_vac),3) if e_vac else None,
        "method": "formation_energy proxy (no supercell DFT)",
        "reference": "Freysoldt et al. Rev Mod Phys 2014",
        "uncertainty": 0.5,
    }
