import os
import json
from pathlib import Path
from typing import Optional

def _cfg_file() -> Path:
    from matgraph.settings import settings
    return settings.config_file

def _cfg_dir() -> Path:
    from matgraph.settings import settings
    return settings.config_dir

def save_api_key(api_key: str):
    f = _cfg_file()
    d = _cfg_dir()
    d.mkdir(parents=True, exist_ok=True)
    config = {}
    if f.exists():
        try:
            with open(f, "r") as fh:
                config = json.load(fh)
        except json.JSONDecodeError:
            pass
    config["mp_api_key"] = api_key
    with open(f, "w") as fh:
        json.dump(config, fh, indent=4)
    try:
        f.chmod(0o600)
    except Exception:
        pass

def get_api_key() -> Optional[str]:
    v = os.environ.get("MP_API_KEY")
    if v:
        return v
    v2 = os.getenv("MATGRAPH_MP_API_KEY")
    if v2:
        return v2
    f = _cfg_file()
    if f.exists():
        try:
            with open(f, "r") as fh:
                return json.load(fh).get("mp_api_key")
        except json.JSONDecodeError:
            return None
    return None

def get_config_value(key: str, default=None):
    """Generic layered get: env MATGRAPH_<UPPER> > config.json."""
    env = os.getenv(f"MATGRAPH_{key.upper()}")
    if env is not None:
        return env
    f = _cfg_file()
    if f.exists():
        try:
            with open(f,"r") as fh:
                return json.load(fh).get(key, default)
        except Exception:
            return default
    return default

def set_config_value(key: str, value):
    f = _cfg_file()
    d = _cfg_dir()
    d.mkdir(parents=True, exist_ok=True)
    cfg = {}
    if f.exists():
        try:
            with open(f,"r") as fh:
                cfg = json.load(fh)
        except Exception:
            pass
    cfg[key] = value
    with open(f,"w") as fh:
        json.dump(cfg, fh, indent=4)
