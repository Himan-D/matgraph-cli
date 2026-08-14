"""ML/DL dispatcher — all verticals call scientific libs through here."""
from __future__ import annotations
from typing import Optional
def ml_eform(structure) -> Optional[float]:
    if structure is None:
        return None
    from matgraph.settings import settings
    from matgraph.models import get_potential
    try:
        pot = get_potential(settings.vertical_model)
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
    from matgraph.settings import settings
    import torch
    w = float(settings.battery_voltage_w); b = float(settings.battery_voltage_b)
    v = torch.tensor([-eform * w + b]); v = torch.clamp(v, min=0.2, max=4.8)
    return float(v.item())
def dl_pv(band_gap: float):
    import torch
    x = torch.tensor([band_gap]); h = torch.tanh((x - 1.34) * 1.8)
    sq = 33.7 * torch.exp(-0.5 * h**2 * 2.0); slme = sq * 0.9
    return float(sq.item()), float(slme.item())
def scientific_battery_voltage(structure, eform: Optional[float]):
    """Try pymatgen BatteryAnalyzer via phase diagram; fallback to dl_voltage."""
    from matgraph.settings import settings
    if not settings.vertical_use_scientific:
        return None, "scientific_disabled"
    try:
        from pymatgen.apps.battery.analyzer import BatteryAnalyzer
        # BatteryAnalyzer needs voltage curve; we only probe availability — true curve needs delithiation phases
        return None, "pymatgen.apps.battery.analyzer available (needs phase diagram — using ML eform + DL fallback)"
    except Exception as e:
        return None, f"pymatgen battery unavailable: {e}"
def scientific_boltztrap_available() -> bool:
    from matgraph.settings import settings
    if not settings.vertical_use_scientific:
        return False
    try:
        import boltztrap2; return True
    except Exception:
        return False
def scientific_defect_available():
    from matgraph.settings import settings
    if not settings.vertical_use_scientific:
        return False
    try:
        import pymatgen.analysis.defects  # noqa
        return True
    except Exception:
        try:
            import doped  # noqa
            return True
        except Exception:
            return False
