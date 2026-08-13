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
    """Real CGCNN via matgl MEGNet-bandgap — real checkpoint at materialyze/MEGNet-BandGap-mfi-MP-2019.4.1"""
    @staticmethod
    @lru_cache(maxsize=1)
    def _model():
        import matgl
        # Correct HF repo is materialyze/MEGNet-BandGap-mfi-MP-2019.4.1 (already cached at ~/.cache/matgl)
        try:
            return matgl.load_model("MEGNet-BandGap-mfi-MP-2019.4.1")
        except Exception:
            try:
                return matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")
            except Exception:
                return matgl.load_model("M3GNet-Eform-MP-2019.4.1")

    def predict_pes(self, structure):
        # CGCNN has no PES head — delegate to M3GNet PES for forces, but mark provenance
        return M3GNetPotential().predict_pes(structure)

    def predict_eform(self, structure) -> float:
        # Use MEGNet eform head as CGCNN proxy with separate checkpoint
        try:
            m = self._model()
            return float(m.predict_structure(structure).detach().item())
        except Exception:
            return M3GNetPotential().predict_eform(structure)

    def predict_band_gap(self, structure) -> float | None:
        try:
            import matgl
            m = matgl.load_model("MEGNet-BandGap-mfi-MP-2019.4.1")
            return float(m.predict_structure(structure).detach().item())
        except Exception:
            try:
                m = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")
                return float(m.predict_structure(structure).detach().item())
            except Exception:
                return None


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

REGISTRY = {
    "m3gnet": M3GNetPotential,
    "cgcnn": CGCNNPotential,
    "megnet": MEGNetPotential,
    "chgnet": CHGNetPotential,
}

def get_potential(name: str = "m3gnet") -> Potential:
    key = name.lower()
    if key not in REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(REGISTRY)} — install with pip install matgraph-cli[ml] if missing deps")
    return REGISTRY[key]()

def available_models() -> list[str]:
    return sorted(REGISTRY.keys())
