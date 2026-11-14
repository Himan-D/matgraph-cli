#!/usr/bin/env python3
"""Generate bundled demo data for `matgraph --demo` (zero-config mode).

Run with the full ML stack (this script needs [ml], the demo itself needs
only the base install)::

    pip install "matgraph-cli[ml]"
    python scripts/make_demo_data.py

What is REAL in the output: crystal structures, composition/volume/density
features, and Cu-Ka XRD patterns — all computed locally with pymatgen from
crystallographic data (no API key needed).

What is SAMPLE: ML numbers (energies, gaps, forces). Running M3GNet here
would bake 100s of MB of weights into this script's runtime for little gain;
every sample value is labeled as such in ``provenance`` and the CLI prints a
DEMO banner. Never present demo output as a prediction.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "matgraph" / "demo_data"

# Illustrative ML-ish numbers. Plausible magnitudes, explicitly labeled samples.
SAMPLES = {
    "LiFePO4": {
        "true_band_gap": 3.50,
        "true_form_energy": -2.45,
        "predicted_band_gap": None,  # m3gnet has no gap head
        "predicted_form_energy": -2.41,
        "m3gnet_energy": -68.20,
    },
    "Si": {
        "true_band_gap": 1.12,
        "true_form_energy": 0.0,  # exact: pure element
        "predicted_band_gap": None,
        "predicted_form_energy": -0.02,
        "m3gnet_energy": -9.10,
    },
    "NaCl": {
        "true_band_gap": 8.50,
        "true_form_energy": -1.55,
        "predicted_band_gap": None,
        "predicted_form_energy": -1.52,
        "m3gnet_energy": -13.30,
    },
}

CRYSTAL_SYSTEM = {"LiFePO4": "Orthorhombic", "Si": "Cubic", "NaCl": "Cubic"}


def build_structures():
    from pymatgen.core import Lattice, Structure

    structs = {}
    # LiFePO4: real olivine structure shipped as a pymatgen fixture.
    import importlib.util
    import os

    _spec = importlib.util.find_spec("pymatgen.util.structures")
    fx = os.path.join(list(_spec.submodule_search_locations)[0], "LiFePO4.json")
    with open(fx) as fh:
        structs["LiFePO4"] = Structure.from_dict(json.load(fh))
    # Si: diamond, Fd-3m, a = 5.431 A (CRC/handbook value).
    structs["Si"] = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.431), ["Si"], [[0, 0, 0]])
    # NaCl: rocksalt, Fm-3m, a = 5.640 A (room-temp value).
    structs["NaCl"] = Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.640), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    return structs


def main() -> None:
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    from pymatgen.io.cif import CifWriter

    from matgraph.verticals.battery import battery_metrics

    OUT.mkdir(parents=True, exist_ok=True)
    structs = build_structures()
    calc = XRDCalculator(wavelength="CuKa")
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    for formula, s in structs.items():
        comp = s.composition
        pattern = calc.get_pattern(s, scaled=True)
        two_theta = [round(float(x), 4) for x in pattern.x.tolist()]
        intensity = [round(float(y), 3) for y in pattern.y.tolist()]
        hkls = [[h["hkl"] for h in group] for group in pattern.hkls]
        smp = SAMPLES[formula]
        # Battery hero metrics: capacity is exact from the formula; voltage is
        # derived from the sample formation energy (labeled sample, not a prediction).
        batt = battery_metrics(
            formula, structure=None, formation_energy_per_atom=smp["true_form_energy"]
        )
        n = len(s)
        # Deterministic tiny sample forces (labeled sample, not a relaxation).
        forces = [[round(0.01 * ((i + a) % 5 - 2), 4) for a in range(3)] for i in range(n)]
        data = {
            "formula": formula,
            "material_id": f"demo-{formula}-1",
            "crystal_system": CRYSTAL_SYSTEM[formula],
            "true_band_gap": smp["true_band_gap"],
            "predicted_band_gap": smp["predicted_band_gap"],
            "band_gap_source": "demo_sample",
            "true_form_energy": smp["true_form_energy"],
            "predicted_form_energy": smp["predicted_form_energy"],
            "m3gnet_energy": smp["m3gnet_energy"],
            "m3gnet_forces": forces,
            "m3gnet_stresses": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "features": {
                "num_elements": len(comp.elements),
                "mean_atomic_mass": round(comp.weight / comp.num_atoms, 4),
                "volume": round(s.volume, 4),
                "density": round(s.density, 4),
            },
            "xrd": {"two_theta": two_theta, "intensity": intensity, "hkls": hkls},
            "structure_cif": str(CifWriter(s)),
            "battery": batt,
            "provenance": {
                "mode": "demo",
                "generated_utc": stamp,
                "computed": ["structure", "features", "xrd", "battery.capacity"],
                "sample": [
                    "true_band_gap",
                    "true_form_energy",
                    "predicted_form_energy",
                    "m3gnet_energy",
                    "m3gnet_forces",
                    "m3gnet_stresses",
                    "battery.voltage",
                ],
                "note": "Bundled zero-config sample. Structure/features/XRD/capacity are computed from real crystallographic data; ML numbers are illustrative samples, NOT predictions.",
            },
        }
        path = OUT / f"{formula.lower()}.json"
        path.write_text(json.dumps(data, indent=1))
        print(f"wrote {path} ({len(two_theta)} XRD peaks)")


if __name__ == "__main__":
    main()
