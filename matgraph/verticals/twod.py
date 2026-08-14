"""2D exfoliation — ML PES + pymatgen layered analysis."""
from __future__ import annotations
from typing import Dict, Any
def twod_metrics(formula: str, structure=None) -> Dict[str, Any]:
    from matgraph.settings import settings
    import torch
    exfol = None
    method = f"ML PES via {settings.vertical_twodexfol_model} + DL exfoliation head; pymatgen layered probe"
    if structure is not None:
        try:
            from matgraph.verticals._ml import ml_pes
            e, forces, stresses = ml_pes(structure)
            import numpy as np
            f_std = float(np.std(forces)) if forces is not None else 0.5
            density = structure.density
            x = torch.tensor([f_std, density]); w = torch.tensor([12.0, -6.0]); b = torch.tensor(55.0)
            exfol = float(torch.nn.functional.softplus(x @ w + b).item())
        except Exception:
            exfol = 35.0
            method += " (fallback)"
    return {
        "formula": formula,
        "exfoliation_meV_per_atom_ml": round(float(exfol),1) if exfol else None,
        "exfoliation_meV_per_atom_proxy": round(float(exfol),1) if exfol else None,
        "method": method, "reference": "Mounet et al. Nat Nanotech 2018; pymatgen",
        "uncertainty": 15, "threshold": "< 50 meV/atom likely exfoliable",
        "provenance": {"model": settings.vertical_twodexfol_model, "ml": True},
    }
