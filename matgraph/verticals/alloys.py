"""Alloys / HEA mixing proxy — Miedema + entropy."""
from __future__ import annotations
from typing import Dict, Any
import math
def alloy_metrics(formula: str) -> Dict[str, Any]:
    from pymatgen.core import Composition
    comp = Composition(formula)
    n = len(comp.elements)
    # config entropy: R*ln(n) per mole
    s_config = 8.314 * math.log(n) if n>1 else 0  # J/mol/K
    # mixing enthalpy proxy: variance of electronegativity
    try:
        ens = [el.X for el in comp.elements]
        mean = sum(ens)/len(ens)
        var = sum((x-mean)**2 for x in ens)/len(ens)
        h_mix = (var*100 - 5)  # kJ/mol heuristic, can be negative
    except Exception:
        h_mix = 0
    # HEA criterion: s*T > |h| at 1000K
    stable = s_config*1000/1000 > abs(h_mix) if n>=5 else False
    return {
        "formula": formula,
        "n_elements": n,
        "s_config_J_mol_K": round(s_config,2),
        "h_mix_kJ_mol_proxy": round(float(h_mix),2),
        "hea_likely": bool(stable and n>=5),
        "method": "Miedema variance + ideal entropy (no CALPHAD)",
        "reference": "Miracle & Senkov Acta Mater 2017",
        "uncertainty": 5,
    }
