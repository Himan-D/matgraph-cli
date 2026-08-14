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
    try:
        f.chmod(0o600)
    except Exception:
        pass

def save_wandb_key(api_key: str, host: str | None = None):
    set_config_value("wandb_api_key", api_key)
    if host:
        set_config_value("wandb_base_url", host)
    # also try to write to wandb's netrc via wandb login for real wandb sync
    try:
        import subprocess, os
        env = os.environ.copy()
        env["WANDB_API_KEY"] = api_key
        if host:
            env["WANDB_BASE_URL"] = host
        subprocess.run(["wandb","login","--relogin"], input=api_key+"\n", text=True, env=env, capture_output=True, timeout=10)
    except Exception:
        pass

def get_wandb_key() -> Optional[str]:
    v = os.environ.get("WANDB_API_KEY")
    if v:
        return v
    v2 = os.getenv("MATGRAPH_WANDB_API_KEY")
    if v2:
        return v2
    return get_config_value("wandb_api_key")

def clear_wandb_key():
    f = _cfg_file()
    if not f.exists():
        return
    try:
        with open(f,"r") as fh:
            cfg = json.load(fh)
        cfg.pop("wandb_api_key", None)
        cfg.pop("wandb_base_url", None)
        with open(f,"w") as fh:
            json.dump(cfg, fh, indent=4)
    except Exception:
        pass
