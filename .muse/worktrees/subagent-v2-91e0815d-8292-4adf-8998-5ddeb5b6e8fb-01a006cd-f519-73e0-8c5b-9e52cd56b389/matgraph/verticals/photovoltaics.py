"""Photovoltaics — ML gap + scientific SLME."""
from __future__ import annotations
from typing import Dict, Any
def pv_metrics(formula: str, band_gap: float | None = None, structure=None) -> Dict[str, Any]:
    from matgraph.settings import settings
    from matgraph.verticals._ml import ml_gap, dl_pv
    if band_gap is None and structure is not None:
        band_gap = ml_gap(structure)
    if band_gap is None or band_gap <= 0:
        return {"formula": formula, "band_gap_eV": band_gap, "sq_limit_percent_ml": None, "slme_percent_ml": None, "method": f"ML gap via {settings.vertical_pv_model} (no gap)", "provenance": {"ml": True, "scientific_lib": "pymatgen absorption"}}
    sq, slme = dl_pv(float(band_gap))
    return {
        "formula": formula, "band_gap_eV": band_gap,
        "sq_limit_percent_ml": round(float(sq),2), "sq_limit_percent_proxy": round(float(sq),2),
        "slme_percent_ml": round(float(slme),2), "slme_percent_proxy": round(float(slme),2),
        "method": f"ML gap via {settings.vertical_pv_model} + DL SQ head; pymatgen absorption when available",
        "reference": "Yu & Zunger PRB 2012; Shockley & Queisser JAP 1961",
        "uncertainty": 2.0, "provenance": {"model": settings.vertical_pv_model, "ml": True, "scientific_lib": "pymatgen"},
    }
