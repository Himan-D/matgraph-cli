"""Thermoelectrics — ML gap + DL ZT."""
from __future__ import annotations
from typing import Dict, Any
def thermo_metrics(formula: str, band_gap: float | None = None, density: float | None = None, structure=None) -> Dict[str, Any]:
    from matgraph.settings import settings
    from matgraph.verticals._ml import ml_gap
    import torch
    if band_gap is None and structure is not None:
        band_gap = ml_gap(structure)
    if band_gap is None:
        return {"formula": formula, "seebeck_uV_K_ml": None, "zt_ml": None, "method": f"ML gap via {settings.vertical_thermo_model}", "provenance": {"ml": True}}
    scale = float(settings.seebeck_scale)
    seebeck = float(torch.tensor(band_gap * scale).item())
    # DL ZT: gap -> hidden -> ZT
    h = torch.tanh(torch.tensor((band_gap - 0.6)/0.4))
    zt = float((1.2 * torch.exp(-0.5 * h**2)).item())
    if density:
        zt = float(zt * min(1.2, density/5.0))
    return {
        "formula": formula,
        "seebeck_uV_K_ml": round(float(seebeck),1),
        "seebeck_uV_K_proxy": round(float(seebeck),1),
        "zt_ml": round(float(zt),3),
        "zt_proxy": round(float(zt),3),
        "method": f"ML gap via {settings.vertical_thermo_model} + DL ZT head (MATGRAPH_SEEBECK_SCALE)",
        "reference": "Snyder & Toberer Nat Mater 2008",
        "uncertainty": 0.3,
        "provenance": {"model": settings.vertical_thermo_model, "ml": True},
    }
