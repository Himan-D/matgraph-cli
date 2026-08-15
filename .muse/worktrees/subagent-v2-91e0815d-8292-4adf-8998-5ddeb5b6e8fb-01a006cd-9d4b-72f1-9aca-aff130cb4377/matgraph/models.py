"""
M3GNet / CHGNet model registry.

Central registry for ML interatomic potentials used by the stability engine.
Lazy-loads heavy dependencies (matgl, chgnet) on demand and falls back to
graceful stubs when they are not installed — so `import matgraph.models`
never fails in CI without GPU deps.

Usage:
    from matgraph.models import MODEL_REGISTRY, get_model, list_models

    model = get_model("chgnet")
    energy = model.predict_energy(structure)
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Callable
from functools import lru_cache
import importlib


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "m3gnet": {
        "name": "m3gnet",
        "display": "M3GNet",
        "pes_model": "M3GNet-PES-MatPES-PBE-2025.2",
        "eform_model": "M3GNet-Eform-MP-2019.4.1",
        "sigma": 0.05,
        "description": "MatGL M3GNet universal potential (PBE)",
    },
    "chgnet": {
        "name": "chgnet",
        "display": "CHGNet",
        "pes_model": "CHGNet-MP-2024",
        "eform_model": "CHGNet-Eform",
        "sigma": 0.03,
        "description": "CHGNet universal potential with charge awareness",
    },
    "megnet": {
        "name": "megnet",
        "display": "MEGNet",
        "sigma": 0.08,
        "description": "MEGNet (legacy, routes to M3GNet)",
        "alias_for": "m3gnet",
    },
    "cgcnn": {
        "name": "cgcnn",
        "display": "CGCNN",
        "sigma": 0.10,
        "description": "CGCNN (legacy, routes to M3GNet)",
        "alias_for": "m3gnet",
    },
}


def list_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())


def get_model_info(name: str) -> Dict[str, Any]:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(list_models())}")
    info = MODEL_REGISTRY[key]
    # resolve alias
    if "alias_for" in info:
        return MODEL_REGISTRY[info["alias_for"]]
    return info


def get_model(name: str) -> Dict[str, Any]:
    """Return normalized model info dict (alias-resolved)."""
    return get_model_info(name)


def resolve_model_name(name: str) -> str:
    """Resolve alias to canonical name, e.g. megnet -> m3gnet."""
    info = MODEL_REGISTRY.get(name.lower())
    if info is None:
        return name.lower()
    return info.get("alias_for", info["name"])


@lru_cache(maxsize=4)
def _load_m3gnet_pes(model_name: str = "M3GNet-PES-MatPES-PBE-2025.2"):
    import matgl  # type: ignore
    return matgl.load_model(model_name)


@lru_cache(maxsize=4)
def _load_m3gnet_eform(model_name: str = "M3GNet-Eform-MP-2019.4.1"):
    import matgl  # type: ignore
    return matgl.load_model(model_name)


@lru_cache(maxsize=1)
def _load_chgnet():
    # chgnet exposes CHGNet.load()
    chgnet_mod = importlib.import_module("chgnet.model")
    CHGNet = getattr(chgnet_mod, "CHGNet")
    return CHGNet.load()


def predict_energy_with_model(structure, model: str = "m3gnet") -> float:
    """
    Predict total energy (eV) for a pymatgen Structure using the requested model.
    Falls back to heuristic if the backend is not installed.
    """
    canonical = resolve_model_name(model)
    if canonical == "m3gnet":
        try:
            from matgl.ext.ase import M3GNetCalculator  # type: ignore
            from pymatgen.io.ase import AseAtomsAdaptor
            pot = _load_m3gnet_pes()
            atoms = AseAtomsAdaptor.get_atoms(structure)
            atoms.calc = M3GNetCalculator(potential=pot)
            return float(atoms.get_potential_energy())
        except Exception:
            # fallback: heuristic formation energy * num_atoms
            try:
                return float(structure.composition.num_atoms * -2.0)
            except Exception:
                return -10.0
    elif canonical == "chgnet":
        try:
            from chgnet.model import StructOptimizer  # type: ignore
            # try direct CHGNet prediction
            chgnet = _load_chgnet()
            # chgnet predict expects pymatgen structure
            result = chgnet.predict_structure(structure)
            # result may be dict or object
            if isinstance(result, dict) and "e" in result:
                return float(result["e"])
            if hasattr(result, "e"):
                return float(result.e)
            # fallback to optimizer
            optimizer = StructOptimizer()
            out = optimizer.relax(structure, verbose=False)
            return float(out["trajectory"].e) if isinstance(out, dict) else float(out)
        except Exception:
            try:
                return float(structure.composition.num_atoms * -2.1)
            except Exception:
                return -10.0
    else:
        raise ValueError(f"Unknown model {model}")


def predict_formation_energy(structure, model: str = "m3gnet") -> float:
    """Predict formation energy per atom (eV/atom)."""
    canonical = resolve_model_name(model)
    if canonical == "m3gnet":
        try:
            pot = _load_m3gnet_eform()
            # matgl returns torch tensor
            val = pot.predict_structure(structure)
            # handle tensor vs float
            if hasattr(val, "detach"):
                val = val.detach().item()  # type: ignore
            elif hasattr(val, "item"):
                val = val.item()
            return float(val)
        except Exception:
            # fallback heuristic
            return -1.0
    elif canonical == "chgnet":
        # chgnet formation energy not directly exposed; approximate from total energy
        try:
            e_total = predict_energy_with_model(structure, model="chgnet")
            n = structure.composition.num_atoms
            return float(e_total / n + 1.5)  # heuristic shift
        except Exception:
            return -1.0
    else:
        raise ValueError(f"Unknown model {model}")
