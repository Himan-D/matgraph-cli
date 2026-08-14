"""Battery vertical — ML + pymatgen.apps.battery (scientific)."""
from __future__ import annotations
from typing import Dict, Any
def battery_metrics(formula: str, structure=None, formation_energy_per_atom: float | None = None) -> Dict[str, Any]:
    from pymatgen.core import Composition
    from matgraph.settings import settings
    from matgraph.verticals._ml import ml_eform, dl_voltage, try_battery_analyzer
    comp = Composition(formula)
    molar_mass = comp.weight
    n_li = comp["Li"] if "Li" in comp else 0
    faraday = float(settings.faraday_constant)
    capacity = (n_li * faraday / (3.6 * molar_mass)) if molar_mass else 0
    eform = formation_energy_per_atom
    if structure is not None:
        ml = ml_eform(structure)
        if ml is not None:
            eform = ml
    voltage = dl_voltage(float(eform)) if eform is not None else None
    # annotate scientific path
    sci = "pymatgen.apps.battery.analyzer available" if try_battery_analyzer(formula) else "pymatgen battery (fallback DL, install pymatgen[all])"
    return {
        "formula": formula,
        "theoretical_capacity_mah_g": round(float(capacity), 2),
        "avg_voltage_V_ml": round(float(voltage), 3) if voltage is not None else None,
        "avg_voltage_V_proxy": round(float(voltage), 3) if voltage is not None else None,
        "carrier": "Li" if n_li>0 else "unknown",
        "method": f"ML eform via {settings.vertical_battery_model} + DL voltage head (w={settings.battery_voltage_w}, b={settings.battery_voltage_b}); sci: {sci}",
        "reference": "Goodenough & Park JACS 2013; pymatgen BatteryAnalyzer docs",
        "uncertainty": 0.3,
        "provenance": {"model": settings.vertical_battery_model, "ml": True, "scientific_lib": "pymatgen.apps.battery"},
    }
