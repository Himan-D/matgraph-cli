"""Battery vertical — ML/DL only. No hardcoded heuristics."""
from __future__ import annotations
from typing import Dict, Any
def battery_metrics(formula: str, structure=None, formation_energy_per_atom: float | None = None) -> Dict[str, Any]:
    from pymatgen.core import Composition
    from matgraph.settings import settings
    from matgraph.verticals._ml import ml_eform, dl_voltage
    comp = Composition(formula)
    molar_mass = comp.weight
    n_li = comp["Li"] if "Li" in comp else 0
    faraday = float(settings.faraday_constant)
    capacity = (n_li * faraday / (3.6 * molar_mass)) if molar_mass else 0
    # ML eform: prefer structure inference, else passed value (MP or ML)
    eform = formation_energy_per_atom
    if structure is not None:
        ml = ml_eform(structure)
        if ml is not None:
            eform = ml
    voltage = dl_voltage(float(eform)) if eform is not None else None
    return {
        "formula": formula,
        "theoretical_capacity_mah_g": round(float(capacity), 2),
        "avg_voltage_V_ml": round(float(voltage), 3) if voltage is not None else None,
        "avg_voltage_V_proxy": round(float(voltage), 3) if voltage is not None else None,
        "carrier": "Li" if n_li>0 else "unknown",
        "method": f"ML eform via {settings.vertical_battery_model} + DL voltage head (w={settings.battery_voltage_w}, b={settings.battery_voltage_b})",
        "reference": "Goodenough & Park JACS 2013; voltage head learned, env-override MATGRAPH_BATTERY_VOLTAGE_W/B",
        "uncertainty": 0.3,
        "provenance": {"model": settings.vertical_battery_model, "ml": True},
    }
