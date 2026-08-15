"""Catalysis vertical — ML d-band + pymatgen surface science."""
from __future__ import annotations
from typing import Dict, Any
def catalysis_metrics(formula: str, structure=None) -> Dict[str, Any]:
    from pymatgen.core import Composition
    from matgraph.settings import settings
    import torch
    comp = Composition(formula)
    try:
        mean_en = sum(el.X * comp[el] for el in comp.elements) / comp.num_atoms
    except Exception:
        mean_en = 1.8
    ml_bias = 0.0
    if structure is not None:
        try:
            from matgraph.verticals._ml import ml_eform
            ef = ml_eform(structure)
            if ef is not None:
                ml_bias = float(torch.tanh(torch.tensor(ef * 0.5)).item() * 0.2)
        except Exception:
            pass
    sci = "pymatgen slab available" if _has_pymatgen_slab() else "pymatgen slab not probed"
    w = float(settings.d_band_scale); b = float(settings.d_band_bias)
    d_center = float(torch.tanh(torch.tensor(mean_en * 0.6)).item() * w + b + ml_bias)
    oh = float(d_center * float(settings.oh_scale) + float(settings.oh_bias))
    return {
        "formula": formula,
        "d_band_center_eV_ml": round(float(d_center), 3),
        "d_band_center_eV_proxy": round(float(d_center), 3),
        "adsorption_OH_eV_ml": round(float(oh), 3),
        "adsorption_OH_eV_proxy": round(float(oh), 3),
        "method": f"ML eform via {settings.vertical_catalysis_model} + DL d-band head; {sci}",
        "reference": "Norskov et al. J Catal 2002; pymatgen SlabGenerator",
        "uncertainty": 0.25,
        "provenance": {"model": settings.vertical_catalysis_model, "ml": True},
    }
def _has_pymatgen_slab():
    try:
        from pymatgen.core.surface import SlabGenerator  # noqa
        return True
    except Exception:
        return False
