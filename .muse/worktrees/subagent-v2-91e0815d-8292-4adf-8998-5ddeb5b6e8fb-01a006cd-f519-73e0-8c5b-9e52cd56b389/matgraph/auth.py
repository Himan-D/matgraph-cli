import json
import secrets
import os
import hashlib
import time
from pathlib import Path
from typing import Optional

def _keys_file() -> Path:
    from matgraph.settings import settings
    return settings.auth_keys_file

def _prefix() -> str:
    from matgraph.settings import settings
    return settings.auth_key_prefix

def load_keys() -> dict:
    f = _keys_file()
    if not f.exists():
        return {}
    with open(f, "r") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            return {}

def save_keys(keys: dict):
    f = _keys_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "w") as fh:
        json.dump(keys, fh, indent=4)
    try:
        f.chmod(0o600)
    except Exception:
        pass

def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()

def generate_api_key(user_name: str, ttl_days: Optional[int] = None, scopes: Optional[list] = None) -> str:
    """Generates a secure API key, stores only hash. No plaintext."""
    from matgraph.settings import settings
    if ttl_days is None:
        ttl_days = settings.auth_default_ttl_days
    if scopes is None:
        scopes = ["read:predict","read:phonon","read:elastic"]
    keys = load_keys()
    raw = _prefix() + secrets.token_urlsafe(24)
    h = _hash_key(raw)
    expires_at = time.time() + ttl_days*86400 if ttl_days else None
    keys[h] = {"user": user_name, "active": True, "scopes": scopes, "created_at": time.time(), "expires_at": expires_at, "prefix": raw[:8]+"..."}
    save_keys(keys)
    return raw

def is_valid_key(api_key: str, required_scope: Optional[str] = None) -> bool:
    master = os.environ.get("MATGRAPH_API_KEY")
    if master and api_key == master:
        return True
    # support legacy plaintext keys file for migration
    h = _hash_key(api_key)
    keys = load_keys()
    # legacy: if keys contain plaintext key directly, migrate check
    if api_key in keys:
        info = keys[api_key]
    else:
        info = keys.get(h)
    if not info or not info.get("active", False):
        return False
    exp = info.get("expires_at")
    if exp and time.time() > exp:
        return False
    if required_scope and required_scope not in info.get("scopes", []):
        return False
    return True

def revoke_key(api_key: str) -> bool:
    h = _hash_key(api_key)
    keys = load_keys()
    # try hash or plaintext
    target = h if h in keys else (api_key if api_key in keys else None)
    if not target:
        return False
    keys[target]["active"] = False
    save_keys(keys)
    return True

def list_keys() -> dict:
    return load_keys()
