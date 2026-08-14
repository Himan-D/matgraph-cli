"""Photovoltaics — SLME proxy from band gap."""
from __future__ import annotations
from typing import Dict, Any
def pv_metrics(formula: str, band_gap: float | None = None) -> Dict[str, Any]:
    # SLME Shockley-Queisser proxy: max at 1.34 eV ~33%, parabolic
    if band_gap is None or band_gap <= 0:
        slme = None
        sq = None
    else:
        # SQ limit heuristic: 33% * exp(-0.5*(gap-1.34)^2)
        import math
        sq = 33.7 * math.exp(-0.5 * ((band_gap - 1.34)/0.5)**2)
        # SLME slightly lower
        slme = sq * 0.9
    return {
        "formula": formula,
        "band_gap_eV": band_gap,
        "sq_limit_percent_proxy": round(float(sq),2) if sq else None,
        "slme_percent_proxy": round(float(slme),2) if slme else None,
        "method": "Shockley-Queisser heuristic from MP band gap (no absorption spectrum)",
        "reference": "Yu & Zunger PRB 2012 (SLME); Shockley & Queisser JAP 1961",
        "uncertainty": 2.0,
    }
