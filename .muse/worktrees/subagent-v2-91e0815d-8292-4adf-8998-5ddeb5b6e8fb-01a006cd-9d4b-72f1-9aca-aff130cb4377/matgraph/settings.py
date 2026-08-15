"""
MatGraph settings — centralized model configuration.

Exposes `pes_model`, `eform_model`, `diffusion_model` as the canonical
configuration knobs for the ML stability engine. Values can be overridden
via environment variables or by mutating the Settings instance.

Primary abstraction: E_hull (energy above convex hull) is the central
stability metric; all other stability views derive from it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    """Global MatGraph configuration."""
    pes_model: str = field(default_factory=lambda: os.environ.get("MATGRAPH_PES_MODEL", "M3GNet-PES-MatPES-PBE-2025.2"))
    eform_model: str = field(default_factory=lambda: os.environ.get("MATGRAPH_EFORM_MODEL", "M3GNet-Eform-MP-2019.4.1"))
    diffusion_model: str = field(default_factory=lambda: os.environ.get("MATGRAPH_DIFFUSION_MODEL", "diffusion-default"))
    default_stability_model: str = field(default_factory=lambda: os.environ.get("MATGRAPH_STABILITY_MODEL", "m3gnet"))
    # uncertainty defaults per model (eV/atom, 1σ)
    model_sigma: dict = field(default_factory=lambda: {
        "m3gnet": 0.05,
        "chgnet": 0.03,
        "megnet": 0.08,
        "cgcnn": 0.10,
    })

    def get_sigma(self, model: str) -> float:
        return self.model_sigma.get(model.lower(), 0.05)


# module-level singletons for backwards-compat imports:
# `from matgraph.settings import pes_model` should work, and
# `matgraph.settings.pes_model` is the canonical string value.
_settings = Settings()

pes_model: str = _settings.pes_model
eform_model: str = _settings.eform_model
diffusion_model: str = _settings.diffusion_model
default_stability_model: str = _settings.default_stability_model

# re-export Settings class and helper
__all__ = ["Settings", "pes_model", "eform_model", "diffusion_model", "default_stability_model", "settings"]

settings = _settings
