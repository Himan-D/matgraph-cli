"""Pluggable potential registry — M3GNet today, CHGNet/ALIGNN tomorrow."""
from __future__ import annotations
import os
from functools import lru_cache
from typing import Protocol

class Potential(Protocol):
    def predict_pes(self, structure): ...
    def predict_eform(self, structure) -> float: ...

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

REGISTRY = {
    "m3gnet": M3GNetPotential,
}

def get_potential(name: str = "m3gnet") -> Potential:
    key = name.lower()
    if key in ("cgcnn","megnet"):
        key = "m3gnet"
    if key not in REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[key]()

def available_models() -> list[str]:
    return sorted(REGISTRY.keys())
