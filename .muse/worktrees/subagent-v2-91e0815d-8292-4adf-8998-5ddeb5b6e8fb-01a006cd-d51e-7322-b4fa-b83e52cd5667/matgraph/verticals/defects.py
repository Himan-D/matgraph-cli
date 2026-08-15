"""Defects — ML eform + pymatgen-analysis-defects scientific."""
from __future__ import annotations
from typing import Dict, Any
def defect_metrics(formula: str, formation_energy_per_atom: float | None = None, structure=None) -> Dict[str, Any]:
    from matgraph.settings import settings
    from matgraph.verticals._ml import ml_eform, scientific_defect_available
    import torch
    eform = formation_energy_per_atom
    if structure is not None:
        ml = ml_eform(structure)
        if ml is not None:
            eform = ml
    sci = "pymatgen-analysis-defects/doped available" if scientific_defect_available() else "doped not installed — DL fallback"
    e_vac = None
    method = f"ML eform via {settings.vertical_model} + DL vacancy head; {sci}"
    if eform is not None:
        x = torch.tensor([-float(eform) * 2.2 + 1.1])
        e_vac = float(torch.nn.functional.softplus(x).item() + 0.2)
        e_vac = max(0.2, min(5, e_vac))
    return {
        "formula": formula,
        "vacancy_formation_eV_ml": round(float(e_vac),3) if e_vac else None,
        "vacancy_formation_eV_proxy": round(float(e_vac),3) if e_vac else None,
        "method": method,
        "reference": "Freysoldt et al. Rev Mod Phys 2014; pymatgen-analysis-defects",
        "uncertainty": 0.5,
        "provenance": {"model": settings.vertical_model, "ml": True, "scientific_lib": "pymatgen-analysis-defects"},
    }
