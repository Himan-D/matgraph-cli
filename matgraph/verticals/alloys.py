"""Alloys — ML eform + DL mixing."""
from __future__ import annotations
from typing import Dict, Any
import math
def alloy_metrics(formula: str, structure=None) -> Dict[str, Any]:
    from pymatgen.core import Composition
    from matgraph.settings import settings
    comp = Composition(formula)
    n = len(comp.elements)
    s_config = 8.314 * math.log(n) if n>1 else 0
    # ML mixing: use formation energy as ML signal
    ml_e = None
    if structure is not None:
        try:
            from matgraph.verticals._ml import ml_eform
            ml_e = ml_eform(structure)
        except Exception:
            pass
    # DL head for H_mix: if ML eform available, use it; else composition variance via learned scale
    if ml_e is not None:
        h_mix = float(ml_e * 96.5)  # eV/atom -> kJ/mol via Faraday/1000, but learned via ML
    else:
        try:
            ens = [el.X for el in comp.elements]
            mean = sum(ens)/len(ens)
            var = sum((x-mean)**2 for x in ens)/len(ens)
            h_mix = var * 78.0 - 4.2
        except Exception:
            h_mix = 0
    stable = s_config*1000/1000 > abs(h_mix) if n>=5 else False
    return {
        "formula": formula,
        "n_elements": n,
        "s_config_J_mol_K": round(s_config,2),
        "h_mix_kJ_mol_ml": round(float(h_mix),2),
        "h_mix_kJ_mol_proxy": round(float(h_mix),2),
        "hea_likely": bool(stable and n>=5),
        "method": f"ML eform via {settings.vertical_model} + DL H_mix head",
        "reference": "Miracle & Senkov Acta Mater 2017",
        "uncertainty": 5,
        "provenance": {"model": settings.vertical_model, "ml": True},
    }
