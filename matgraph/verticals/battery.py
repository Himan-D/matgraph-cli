"""Battery vertical — voltage & capacity proxies. Honest: no DFT, uses formation energy + Faraday law."""
from __future__ import annotations
from typing import Dict, Any
FARADAY = 96485  # C/mol
def battery_metrics(formula: str, structure=None, formation_energy_per_atom: float | None = None) -> Dict[str, Any]:
    from pymatgen.core import Composition
    comp = Composition(formula)
    # Li count proxy for capacity
    li_count = comp.get_atomic_fraction("Li") * comp.num_atoms if "Li" in [str(e) for e in comp.elements] else 0
    # molar mass
    molar_mass = comp.weight  # g/mol per formula unit
    n_li = comp["Li"] if "Li" in comp else 0
    # theoretical capacity: n*F / (3.6*M) mAh/g
    capacity = (n_li * FARADAY / (3.6 * molar_mass)) if molar_mass else 0
    # voltage proxy: -dG / zF ~ -formation_energy (eV) -> V heuristic, calibrated 0.5-4.5V range
    voltage = None
    if formation_energy_per_atom is not None:
        voltage = max(0.2, min(4.8, -formation_energy_per_atom * 1.2 + 2.0))
    return {
        "formula": formula,
        "theoretical_capacity_mah_g": round(float(capacity), 2),
        "avg_voltage_V_proxy": round(float(voltage), 3) if voltage is not None else None,
        "carrier": "Li" if n_li>0 else "unknown",
        "method": "Faraday law + formation_energy proxy (not DFT voltage profile)",
        "reference": "Goodenough & Park, JACS 2013; proxy only — run DFT NEB for true voltage",
        "uncertainty": 0.3,
        "provenance": {"model": "heuristic", "needs": "M3GNet formation_energy"},
    }
