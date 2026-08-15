"""Pluggable potential registry — M3GNet + MEGNet + CGCNN + CHGNet (MatGL FMMs)."""
from __future__ import annotations
import os
from functools import lru_cache
from typing import Protocol

class Potential(Protocol):
    def predict_pes(self, structure): ...
    def predict_eform(self, structure) -> float: ...
    def predict_band_gap(self, structure) -> float | None: ...

class M3GNetPotential:
    @property
    def pes_name(self) -> str:
        from matgraph.settings import settings
        return settings.pes_model
    @property
    def eform_name(self) -> str:
        from matgraph.settings import settings
        return settings.eform_model

    @staticmethod
    @lru_cache(maxsize=1)
    def _pes():
        import matgl
        from matgraph.settings import settings
        return matgl.load_model(settings.pes_model)

    @staticmethod
    @lru_cache(maxsize=1)
    def _eform():
        import matgl
        from matgraph.settings import settings
        return matgl.load_model(settings.eform_model)

    def predict_pes(self, structure):
        from matgl.ext.ase import M3GNetCalculator
        from pymatgen.io.ase import AseAtomsAdaptor
        pot = self._pes()
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = M3GNetCalculator(potential=pot)
        return atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()

    def predict_eform(self, structure) -> float:
        m = self._eform()
        return float(m.predict_structure(structure).detach().item())

    def predict_band_gap(self, structure) -> float | None:
        return None  # M3GNet has no band_gap head


class CGCNNPotential:
    """CGCNN — honest: requires DGL + txie-93/cgcnn checkpoint; no MEGNet proxy."""
    def predict_pes(self, structure):
        from matgraph.exceptions import ModelInferenceError
        raise ModelInferenceError("CGCNN PES not available — CGCNN has no PES head and no M3GNet fallback is allowed. Install CGCNN via DGL + txie-93/cgcnn and implement CGCNNPotential.predict_pes, or use --model m3gnet/chgnet.")

    def predict_eform(self, structure) -> float:
        from matgraph.exceptions import ModelInferenceError
        # Do not proxy MEGNet — fail honestly per audit
        try:
            import dgl  # type: ignore
            import cgcnn  # type: ignore
            # Real path: cgcnn.model.CGCNN.load("txie-93/cgcnn")
            raise ModelInferenceError("CGCNN checkpoint txie-93/cgcnn not wired — implement DGL CGCNN loading here")
        except ModelInferenceError:
            raise
        except Exception as e:
            raise ModelInferenceError(f"CGCNN not installed — pip install dgl + cgcnn checkpoint: {e}") from e

    def predict_band_gap(self, structure) -> float | None:
        from matgraph.exceptions import ModelInferenceError
        try:
            import dgl  # type: ignore
            import cgcnn  # type: ignore
            raise ModelInferenceError("CGCNN band_gap checkpoint not wired")
        except ModelInferenceError:
            raise
        except Exception as e:
            raise ModelInferenceError(f"CGCNN not installed: {e}") from e


class MEGNetPotential:
    @staticmethod
    @lru_cache(maxsize=1)
    def _eform():
        import matgl
        try:
            return matgl.load_model("MEGNet-Eform-MP-2019.4.1")
        except Exception:
            return matgl.load_model("MEGNet-MP-2019.4.1-Eform")

    @staticmethod
    @lru_cache(maxsize=1)
    def _bandgap():
        import matgl
        try:
            return matgl.load_model("MEGNet-BandGap-mfi-MP-2019.4.1")
        except Exception:
            try:
                return matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")
            except Exception:
                return None

    def predict_pes(self, structure):
        return M3GNetPotential().predict_pes(structure)

    def predict_eform(self, structure) -> float:
        return float(self._eform().predict_structure(structure).detach().item())

    def predict_band_gap(self, structure) -> float | None:
        m = self._bandgap()
        if m is None:
            return None
        try:
            return float(m.predict_structure(structure).detach().item())
        except Exception:
            return None

class CHGNetPotential:
    """CHGNet FMM via MatGL — materialyze/CHGNet-MP-2024.2.13 or fallback."""
    @staticmethod
    @lru_cache(maxsize=1)
    def _pes():
        import matgl
        for name in ["CHGNet-MP-2024.2.13", "CHGNet-MP-2023.9.1", "M3GNet-PES-MatPES-PBE-2025.2"]:
            try:
                return matgl.load_model(name)
            except Exception:
                continue
        raise RuntimeError("No CHGNet/M3GNet PES available — pip install matgraph-cli[ml]")

    def predict_pes(self, structure):
        from matgl.ext.ase import M3GNetCalculator
        from pymatgen.io.ase import AseAtomsAdaptor
        # CHGNet uses same ASE calculator interface via matgl
        try:
            import matgl.ext.ase as ext
            # try CHGNetCalculator if present
            calc_cls = getattr(ext, "CHGNetCalculator", None) or getattr(ext, "M3GNetCalculator")
        except Exception:
            from matgl.ext.ase import M3GNetCalculator as calc_cls
        pot = self._pes()
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = calc_cls(potential=pot)
        return atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()

    def predict_eform(self, structure) -> float:
        # CHGNet is PES-only; use M3GNet Eform head for formation energy
        return M3GNetPotential().predict_eform(structure)

    def predict_band_gap(self, structure) -> float | None:
        return None

class OMat24Potential:
    """OMat24 EquiformerV2 — requires fairchem-core + checkpoint; no M3GNet fallback."""

    @staticmethod
    @lru_cache(maxsize=1)
    def _fairchem():
        try:
            from fairchem.core.models.equiformer_v2 import EquiformerV2  # type: ignore
            return EquiformerV2
        except Exception:
            return None

    def predict_pes(self, structure):
        from matgraph.exceptions import ModelInferenceError
        Fc = self._fairchem()
        if Fc is None:
            raise ModelInferenceError("OMat24 not installed — pip install matgraph-cli[omat24] (fairchem-core) + OMat24 checkpoint. Refusing M3GNet fallback per audit.")
        # Real inference must load checkpoint: Fc.from_pretrained("facebook/OMat24-EquiformerV2")
        raise ModelInferenceError("OMat24 checkpoint/runtime not configured — set MATGRAPH_OMAT24_CHECKPOINT and wire EquiformerV2.from_pretrained")

    def predict_eform(self, structure) -> float:
        from matgraph.exceptions import ModelInferenceError
        raise ModelInferenceError("OMat24 eform not available via M3GNet fallback — fix per audit. Use --model m3gnet/megnet/chgnet.")

    def predict_band_gap(self, structure) -> float | None:
        return None


REGISTRY = {
    "m3gnet": M3GNetPotential,
    "cgcnn": CGCNNPotential,
    "megnet": MEGNetPotential,
    "chgnet": CHGNetPotential,
    "omat24": OMat24Potential,
    "equiformer": OMat24Potential,
}

def get_potential(name: str = "m3gnet") -> Potential:
    key = name.lower()
    if key not in REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(REGISTRY)} — install with pip install matgraph-cli[ml] if missing deps")
    return REGISTRY[key]()

def available_models(include_unavailable: bool = True) -> list[str]:
    if include_unavailable:
        return sorted(REGISTRY.keys())
    # Hide unavailable: cgcnn needs dgl, omat24 needs fairchem
    avail = ["m3gnet","megnet","chgnet"]
    try:
        import dgl  # noqa
        avail.append("cgcnn")
    except Exception:
        pass
    try:
        from fairchem.core.models.equiformer_v2 import EquiformerV2  # noqa
        avail.extend(["omat24","equiformer"])
    except Exception:
        pass
    return sorted(set(avail))
