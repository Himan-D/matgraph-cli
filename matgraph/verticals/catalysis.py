"""Catalysis vertical — adsorption & d-band proxy."""
from __future__ import annotations
from typing import Dict, Any
def catalysis_metrics(formula: str, structure=None) -> Dict[str, Any]:
    from pymatgen.core import Composition
    # heuristic d-band center from electronegativity & element d-count
    comp = Composition(formula)
    # simple descriptor: mean electronegativity * valence electron proxy
    try:
        mean_en = sum(el.X * comp[el] for el in comp.elements) / comp.num_atoms
    except Exception:
        mean_en = 1.8
    # d-band center proxy: -mean_en * 0.8 (more electronegative -> deeper)
    d_center = -mean_en * 0.8
    # adsorption proxy for *OH (eV): linear with d_center, Norskov 2007 trend
    e_oh = d_center * 0.5 - 1.2
    return {
        "formula": formula,
        "d_band_center_eV_proxy": round(float(d_center), 3),
        "adsorption_OH_eV_proxy": round(float(e_oh), 3),
        "method": "Norskov d-band heuristic (no DFT slab)",
        "reference": "Norskov et al. J Catal 2002; Hammer & Norskov Nature 1995",
        "uncertainty": 0.25,
        "note": "True catalysis needs slab + site-specific DFT (VASP/QE). Use matgraph dft for bridge.",
    }
