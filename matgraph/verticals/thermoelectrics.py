"""Thermoelectrics — ZT proxy via band gap + mass."""
from __future__ import annotations
from typing import Dict, Any
def thermo_metrics(formula: str, band_gap: float | None = None, density: float | None = None) -> Dict[str, Any]:
    # proxy: narrow gap + heavy mass -> better ZT (heuristic)
    zt_proxy = None
    seebeck = None
    if band_gap is not None:
        # Seebeck proxy: ~ band_gap * 200 uV/K heuristic
        seebeck = band_gap * 150
        # ZT proxy: peaks ~0.5-1 eV
        import math
        zt_proxy = 1.2 * math.exp(-0.5*((band_gap-0.6)/0.4)**2)
        if density:
            zt_proxy *= min(1.2, density/5.0)
    return {
        "formula": formula,
        "seebeck_uV_K_proxy": round(float(seebeck),1) if seebeck else None,
        "zt_proxy": round(float(zt_proxy),3) if zt_proxy else None,
        "method": "Mott heuristic (no BoltzTraP)",
        "reference": "Snyder & Toberer Nat Mater 2008",
        "uncertainty": 0.3,
    }
