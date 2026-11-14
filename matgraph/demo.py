"""Zero-config demo mode — works on the slim base install, no API key, no ML stack.

``matgraph predict Si --demo`` returns bundled sample output so a new user goes
from ``pip install`` to a first successful command in under 60 seconds.

Honesty rules (this module must never violate them):
- Structures, composition/volume/density features, and XRD patterns are REAL:
  computed with pymatgen from crystallographic data (see
  ``scripts/make_demo_data.py``) and frozen with provenance.
- ML numbers (energies, gaps, forces) are ILLUSTRATIVE SAMPLES. Every demo
  row carries ``provenance.mode == "demo"`` and the CLI prints a DEMO banner.
  Never present demo output as a prediction.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from matgraph.exceptions import DataNotFoundError, ValidationError

DATA_DIR = Path(__file__).resolve().parent / "demo_data"

DEMO_FORMULAS = ("LiFePO4", "Si", "NaCl")

BANNER = (
    "[bold yellow]DEMO MODE[/bold yellow] — bundled sample output "
    "(real structure/XRD, illustrative ML numbers). "
    'For live predictions: pip install "matgraph-cli[ml]" + MP_API_KEY.'
)

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def _canonical(formula: str) -> str:
    for cand in DEMO_FORMULAS:
        if cand.lower() == formula.lower():
            return cand
    raise DataNotFoundError(
        f"No demo data for '{formula}'. Demo formulas: {', '.join(DEMO_FORMULAS)}. "
        'For live predictions: pip install "matgraph-cli[ml]" and set MP_API_KEY.'
    )


def _load(formula: str) -> dict[str, Any]:
    formula = _canonical(formula)
    path = DATA_DIR / f"{formula.lower()}.json"
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise DataNotFoundError(f"Demo data file missing: {path}") from None


def demo_predict(formula: str, model: str = "m3gnet") -> list[dict[str, Any]]:
    """Sample ``run_pipeline``-shaped rows (no Structure object — see ``structure_cif``)."""
    from matgraph.schemas import PredictRequest

    formula = _canonical(formula)
    try:
        PredictRequest(formula=formula, model=model)
    except Exception as e:
        raise ValidationError(str(e)) from e
    d = _load(formula)
    low = model.lower()
    pred_gap = d["predicted_band_gap"]
    source = d["band_gap_source"]
    if low in ("megnet", "cgcnn") and pred_gap is None:
        # Illustrative gap-head sample for gap-predicting models (still labeled demo).
        pred_gap = round(d["true_band_gap"] * 0.97, 3)
        source = "demo_sample_gap_head"
    row = {
        "material_id": d["material_id"],
        "formula": d["formula"],
        "true_band_gap": d["true_band_gap"],
        "predicted_band_gap": pred_gap,
        "band_gap_source": source,
        "band_gap_note": "Demo sample — not a prediction.",
        "true_form_energy": d["true_form_energy"],
        "predicted_form_energy": d["predicted_form_energy"],
        "m3gnet_energy": d["m3gnet_energy"],
        "m3gnet_forces": copy.deepcopy(d["m3gnet_forces"]),
        "m3gnet_stresses": copy.deepcopy(d["m3gnet_stresses"]),
        "crystal_system": d["crystal_system"],
        "features": dict(d["features"]),
        "model_used": model.upper() + "-DEMO",
        "provenance": dict(d["provenance"]),
        "structure": None,  # use structure_cif below (needs [mp] to parse)
        "structure_cif": d["structure_cif"],
    }
    return [row]


def demo_xrd(formula: str) -> dict[str, Any]:
    """REAL Cu-Ka XRD pattern (computed with pymatgen, frozen with provenance)."""
    d = _load(formula)
    return {
        "formula": d["formula"],
        "material_id": d["material_id"],
        "two_theta": list(d["xrd"]["two_theta"]),
        "intensity": list(d["xrd"]["intensity"]),
        "hkls": copy.deepcopy(d["xrd"]["hkls"]),
        "provenance": dict(d["provenance"]),
    }


def demo_battery(formula: str) -> dict[str, Any]:
    """Sample battery metrics: capacity is computed from the formula (exact);
    voltage is derived from the sample formation energy (labeled sample)."""
    d = _load(formula)
    out = dict(d["battery"])
    out["provenance"] = dict(d["provenance"])
    return out


def _parse_tokens(formula: str) -> list[str]:
    if not formula or not re.fullmatch(r"([A-Z][a-z]?\d*)+", formula):
        raise ValidationError(f"Invalid formula '{formula}'")
    return [m.group(1) for m in _FORMULA_TOKEN.finditer(formula)]


def demo_substitute(formula: str, element_out: str, element_in: str) -> dict[str, Any]:
    """Sample substitution analysis. Energies are illustrative samples."""
    if element_out == element_in:
        raise ValidationError("element_out and element_in must differ")
    tokens = _parse_tokens(formula)
    if element_out not in tokens:
        raise ValidationError(f"Element {element_out} not found in {formula}")
    d = _load(formula)
    # Deterministic illustrative delta (labeled sample): +0.18 eV per replaced atom.
    n_replaced = tokens.count(element_out)
    orig_e = d["m3gnet_energy"]
    hypo_e = round(orig_e + 0.18 * n_replaced, 3)
    hypo_formula = re.sub(rf"{re.escape(element_out)}(\d*)", rf"{element_in}\1", formula, count=1)
    return {
        "original": {"formula": formula, "energy": orig_e, "max_force": 0.02},
        "hypothetical": {"formula": hypo_formula, "energy": hypo_e, "max_force": 0.03},
        "is_more_stable": bool(hypo_e < orig_e),
        "provenance": dict(d["provenance"]),
    }
