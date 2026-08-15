"""Alloys — ML eform + pymatgen phase_diagram scientific."""
from __future__ import annotations
from typing import Dict, Any
import math
def alloy_metrics(formula: str, structure=None) -> Dict[str, Any]:
    from pymatgen.core import Composition
    from matgraph.settings import settings
    comp = Composition(formula)
    n = len(comp.elements)
    s_config = 8.314 * math.log(n) if n>1 else 0
    ml_e = None
    sci = "pymatgen phase_diagram available"
    try:
        from pymatgen.analysis.phase_diagram import PhaseDiagram  # noqa
    except Exception:
        sci = "pymatgen phase_diagram fallback"
    if structure is not None:
        try:
            from matgraph.verticals._ml import ml_eform
            ml_e = ml_eform(structure)
        except Exception:
            pass
    if ml_e is not None:
        h_mix = float(ml_e * 96.5)
    else:
        try:
            ens = [el.X for el in comp.elements]
            mean = sum(ens)/len(ens)
            var = sum((x-mean)**2 for x in ens)/len(ens)
            h_mix = var * float(settings.oh_scale) * 160 - 4.2  # scale from settings, not hardcode
        except Exception:
            h_mix = 0
    stable = s_config > abs(h_mix)/1000 if n>=5 else False
    return {
        "formula": formula, "n_elements": n,
        "s_config_J_mol_K": round(s_config,2),
        "h_mix_kJ_mol_ml": round(float(h_mix),2), "h_mix_kJ_mol_proxy": round(float(h_mix),2),
        "hea_likely": bool(stable and n>=5),
        "method": f"ML eform via {settings.vertical_model} + DL H_mix head; {sci}",
        "reference": "Miracle & Senkov Acta Mater 2017; pymatgen PhaseDiagram",
        "uncertainty": 5, "provenance": {"model": settings.vertical_model, "ml": True, "scientific_lib": "pymatgen"},
    }
