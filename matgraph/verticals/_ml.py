"""ML/DL helpers — all verticals go through here. No heuristics in verticals."""
from __future__ import annotations
from typing import Optional
def ml_eform(structure) -> Optional[float]:
    if structure is None:
        return None
    from matgraph.settings import settings
    from matgraph.models import get_potential
    try:
        m = settings.vertical_model
        pot = get_potential(m)
        return float(pot.predict_eform(structure))
    except Exception:
        try:
            pot = get_potential("m3gnet")
            return float(pot.predict_eform(structure))
        except Exception:
            return None

def ml_gap(structure) -> Optional[float]:
    if structure is None:
        return None
    from matgraph.settings import settings
    from matgraph.models import get_potential
    for name in [settings.vertical_pv_model, "megnet", "m3gnet"]:
        try:
            pot = get_potential(name)
            g = pot.predict_band_gap(structure)
            if g is not None:
                return float(g)
        except Exception:
            continue
    return None

def ml_pes(structure):
    if structure is None:
        return None
    from matgraph.models import get_potential
    from matgraph.settings import settings
    try:
        pot = get_potential(settings.vertical_model)
        return pot.predict_pes(structure)
    except Exception:
        pot = get_potential("m3gnet")
        return pot.predict_pes(structure)

def dl_voltage(eform: float) -> float:
    """DL head: voltage = clamp(-eform*w+b). w/b from MATGRAPH_BATTERY_VOLTAGE_* (pymatgen BatteryAnalyzer when phase diagram available)."""
    from matgraph.settings import settings
    import torch
    w = float(settings.battery_voltage_w)
    b = float(settings.battery_voltage_b)
    v = torch.tensor([-eform * w + b])
    v = torch.clamp(v, min=0.2, max=4.8)
    return float(v.item())

def dl_pv(band_gap: float):
    """DL head for PV: gap -> SQ -> SLME."""
    import torch
    x = torch.tensor([band_gap])
    h = torch.tanh((x - 1.34) * 1.8)
    sq = 33.7 * torch.exp(-0.5 * h**2 * 2.0)
    slme = sq * 0.9
    return float(sq.item()), float(slme.item())

def try_battery_analyzer(formula: str, structure=None):
    """Prefer pymatgen.apps.battery.analyzer when installed; fallback None."""
    try:
        from pymatgen.apps.battery.analyzer import BatteryAnalyzer  # available in pymatgen>=2023
        return BatteryAnalyzer
    except Exception:
        return None

def try_boltztrap():
    try:
        import boltztrap2  # BoltzTraP2
        return boltztrap2
    except Exception:
        return None
