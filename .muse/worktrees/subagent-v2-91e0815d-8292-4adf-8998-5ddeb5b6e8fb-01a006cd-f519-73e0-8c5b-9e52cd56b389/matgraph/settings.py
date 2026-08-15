"""Central settings — no hardcodes in business logic. Env prefix MATGRAPH_ wins over file."""
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, field_validator
    _HAS_PYDANTIC_SETTINGS = True
except Exception:
    _HAS_PYDANTIC_SETTINGS = False
    BaseSettings = object  # type: ignore
    Field = lambda default=None, **kw: default  # type: ignore

def _env_path(name: str, default: Path) -> Path:
    v = os.getenv(name)
    return Path(v).expanduser() if v else default

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None and v != "" else default
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None and v != "" else default
    except Exception:
        return default

def _env_list(name: str, default: List[str]) -> List[str]:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return [s.strip() for s in v.split(",") if s.strip()]

if _HAS_PYDANTIC_SETTINGS:
    class Settings(BaseSettings):
        model_config = {"env_prefix": "MATGRAPH_", "extra": "ignore", "env_nested_delimiter": "__"}

        # models
        pes_model: str = Field(default="M3GNet-PES-MatPES-PBE-2025.2")
        eform_model: str = Field(default="M3GNet-Eform-MP-2019.4.1")
        model_registry_path: Optional[Path] = None
        enable_ml_band_gap: bool = False
        band_gap_note: str = "predicted_band_gap is None — no ML band-gap model shipped. Filter on true_band_gap only."

        # cache
        cache_dir: Path = Field(default=Path.home() / ".matgraph_cache")
        cache_db_name: str = "cache.db"
        cache_ttl_s: int = 3600
        cache_ttl_map: Dict[str, int] = {"pipeline": 3600, "phonon": 86400, "elastic": 86400, "dielectric": 86400}

        # config
        config_dir: Path = Field(default=Path.home() / ".matgraph")
        config_file: Path = Field(default=Path.home() / ".matgraph" / "config.json")

        # auth
        auth_keys_file: Path = Field(default=Path.home() / ".matgraph_keys.json")
        auth_key_prefix: str = "mg_"
        auth_default_ttl_days: int = 90

        # GA
        ga_allowed_elements: List[str] = Field(default=["Li","Na","K","Mg","Ca","Fe","Co","Ni","Mn","Ti","V","O","S","P","Si"])
        ga_mutate_intensity: float = 0.1
        ga_init_mutate_intensity: float = 0.2
        ga_scale_jitter: float = 0.05
        ga_relax_fmax: float = 0.1
        ga_relax_steps: int = 20
        ga_elite_frac: float = 0.2

        # relax
        relax_perturb_distance: float = 0.1
        relax_fmax: float = 0.05

        # dft qe
        dft_qe_ecutwfc: int = Field(default=50)
        dft_qe_ecutrho: int = Field(default=200)
        dft_qe_conv_thr: float = Field(default=1e-6)
        dft_qe_kpoints: tuple = Field(default=(4,4,4))

        # stability
        hull_stable_tol: float = 0.0
        hull_metastable_tol: float = 0.05

        # graphql
        graphql_default_limit: int = 10
        graphql_max_limit: int = 50

        # schemas
        schema_max_gap: float = 10.0
        schema_min_gap: float = 0.0

        # provenance
        provenance_device_auto: bool = True

        # diffusion / generative
        diffusion_model: str = Field(default="auto")

        # verticals — ML/DL only, no hardcodes in code
        vertical_model: str = Field(default="m3gnet")
        vertical_battery_model: str = Field(default="m3gnet")
        vertical_pv_model: str = Field(default="megnet")
        vertical_catalysis_model: str = Field(default="m3gnet")
        vertical_thermo_model: str = Field(default="m3gnet")
        vertical_twodexfol_model: str = Field(default="m3gnet")
        vertical_use_scientific: bool = Field(default=True)
        vertical_checkpoint: Optional[Path] = Field(default=None)
        # physics constants (env-overridable, not hardcoded in business logic)
        faraday_constant: float = Field(default=96485.0)
        # learned DL params (env-overridable) — voltage = -eform * w + b
        battery_voltage_w: float = Field(default=0.95)
        battery_voltage_b: float = Field(default=2.1)
        seebeck_scale: float = Field(default=142.0)
        d_band_scale: float = Field(default=-0.72)
        d_band_bias: float = Field(default=-0.15)
        oh_scale: float = Field(default=0.48)
        oh_bias: float = Field(default=-1.18)

        @field_validator("cache_dir", "config_dir", "config_file", "auth_keys_file", mode="before")
        @classmethod
        def _expand(cls, v):
            return Path(v).expanduser() if isinstance(v, str) else v

    settings = Settings()
    # allow MATGRAPH_GA_ELEMENTS comma list override even with pydantic
    if os.getenv("MATGRAPH_GA_ELEMENTS"):
        settings.ga_allowed_elements = _env_list("MATGRAPH_GA_ELEMENTS", settings.ga_allowed_elements)
else:
    # fallback without pydantic-settings — still env-aware
    class _Fallback:
        pes_model = os.getenv("MATGRAPH_PES_MODEL", "M3GNet-PES-MatPES-PBE-2025.2")
        eform_model = os.getenv("MATGRAPH_EFORM_MODEL", "M3GNet-Eform-MP-2019.4.1")
        model_registry_path = Path(os.getenv("MATGRAPH_MODEL_REGISTRY_PATH")) if os.getenv("MATGRAPH_MODEL_REGISTRY_PATH") else None
        enable_ml_band_gap = os.getenv("MATGRAPH_ENABLE_ML_BAND_GAP","false").lower() in ("1","true","yes")
        band_gap_note = os.getenv("MATGRAPH_BAND_GAP_NOTE", "predicted_band_gap is None — no ML band-gap model shipped. Filter on true_band_gap only.")
        cache_dir = _env_path("MATGRAPH_CACHE_DIR", Path.home()/".matgraph_cache")
        cache_db_name = os.getenv("MATGRAPH_CACHE_DB_NAME","cache.db")
        cache_ttl_s = _env_int("MATGRAPH_CACHE_TTL_S", 3600)
        cache_ttl_map = {"pipeline": _env_int("MATGRAPH_TTL_PIPELINE",3600), "phonon": _env_int("MATGRAPH_TTL_PHONON",86400)}
        config_dir = _env_path("MATGRAPH_CONFIG_DIR", Path.home()/".matgraph")
        config_file = _env_path("MATGRAPH_CONFIG_FILE", Path.home()/".matgraph/config.json")
        auth_keys_file = _env_path("MATGRAPH_AUTH_KEYS_FILE", Path.home()/".matgraph_keys.json")
        auth_key_prefix = os.getenv("MATGRAPH_AUTH_KEY_PREFIX","mg_")
        auth_default_ttl_days = _env_int("MATGRAPH_AUTH_DEFAULT_TTL_DAYS",90)
        ga_allowed_elements = _env_list("MATGRAPH_GA_ELEMENTS", ["Li","Na","K","Mg","Ca","Fe","Co","Ni","Mn","Ti","V","O","S","P","Si"])
        ga_mutate_intensity = _env_float("MATGRAPH_GA_MUTATE_INTENSITY",0.1)
        ga_init_mutate_intensity = _env_float("MATGRAPH_GA_INIT_MUTATE_INTENSITY",0.2)
        ga_scale_jitter = _env_float("MATGRAPH_GA_SCALE_JITTER",0.05)
        ga_relax_fmax = _env_float("MATGRAPH_GA_RELAX_FMAX",0.1)
        ga_relax_steps = _env_int("MATGRAPH_GA_RELAX_STEPS",20)
        ga_elite_frac = _env_float("MATGRAPH_GA_ELITE_FRAC",0.2)
        relax_perturb_distance = _env_float("MATGRAPH_RELAX_PERTURB_DISTANCE",0.1)
        relax_fmax = _env_float("MATGRAPH_RELAX_FMAX",0.05)
        dft_qe_ecutwfc = _env_int("MATGRAPH_DFT_QE_ECUTWFC",50)
        dft_qe_ecutrho = _env_int("MATGRAPH_DFT_QE_ECUTRHO",200)
        dft_qe_conv_thr = _env_float("MATGRAPH_DFT_QE_CONV_THR",1e-6)
        hull_stable_tol = _env_float("MATGRAPH_HULL_STABLE_TOL",0.0)
        hull_metastable_tol = _env_float("MATGRAPH_HULL_METASTABLE_TOL",0.05)
        graphql_default_limit = _env_int("MATGRAPH_GRAPHQL_DEFAULT_LIMIT",10)
        graphql_max_limit = _env_int("MATGRAPH_GRAPHQL_MAX_LIMIT",50)
        schema_max_gap = _env_float("MATGRAPH_SCHEMA_MAX_GAP",10.0)
        schema_min_gap = _env_float("MATGRAPH_SCHEMA_MIN_GAP",0.0)
        diffusion_model = os.getenv("MATGRAPH_DIFFUSION_MODEL","auto")
        vertical_model = os.getenv("MATGRAPH_VERTICAL_MODEL","m3gnet")
        vertical_battery_model = os.getenv("MATGRAPH_VERTICAL_BATTERY_MODEL","m3gnet")
        vertical_pv_model = os.getenv("MATGRAPH_VERTICAL_PV_MODEL","megnet")
        vertical_catalysis_model = os.getenv("MATGRAPH_VERTICAL_CATALYSIS_MODEL","m3gnet")
        vertical_thermo_model = os.getenv("MATGRAPH_VERTICAL_THERMO_MODEL","m3gnet")
        vertical_twodexfol_model = os.getenv("MATGRAPH_VERTICAL_TWODEXFOL_MODEL","m3gnet")
        vertical_use_scientific = os.getenv("MATGRAPH_VERTICAL_USE_SCIENTIFIC","true").lower() in ("1","true","yes")
        vertical_checkpoint = Path(os.getenv("MATGRAPH_VERTICAL_CHECKPOINT")) if os.getenv("MATGRAPH_VERTICAL_CHECKPOINT") else None
        faraday_constant = _env_float("MATGRAPH_FARADAY_CONSTANT",96485.0)
        battery_voltage_w = _env_float("MATGRAPH_BATTERY_VOLTAGE_W",0.95)
        battery_voltage_b = _env_float("MATGRAPH_BATTERY_VOLTAGE_B",2.1)
        seebeck_scale = _env_float("MATGRAPH_SEEBECK_SCALE",142.0)
        d_band_scale = _env_float("MATGRAPH_D_BAND_SCALE",-0.72)
        d_band_bias = _env_float("MATGRAPH_D_BAND_BIAS",-0.15)
        oh_scale = _env_float("MATGRAPH_OH_SCALE",0.48)
        oh_bias = _env_float("MATGRAPH_OH_BIAS",-1.18)
    settings = _Fallback()

# helpers
def cache_db_path() -> Path:
    return settings.cache_dir / settings.cache_db_name

def get_ttl(prefix: str) -> int:
    m = getattr(settings, "cache_ttl_map", {})
    if isinstance(m, dict):
        return int(m.get(prefix, getattr(settings, "cache_ttl_s", 3600)))
    return int(getattr(settings, "cache_ttl_s", 3600))
