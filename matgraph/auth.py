import json
import secrets
import os
from pathlib import Path
from typing import Optional

KEYS_FILE = Path.home() / ".matgraph_keys.json"

def load_keys() -> dict:
    if not KEYS_FILE.exists():
        return {}
    with open(KEYS_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_keys(keys: dict):
    with open(KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=4)

def generate_api_key(user_name: str) -> str:
    """Generates a secure API key for a user and saves it."""
    keys = load_keys()
    new_key = "mg_" + secrets.token_urlsafe(24)
    keys[new_key] = {
        "user": user_name,
        "active": True
    }
    save_keys(keys)
    return new_key

def is_valid_key(api_key: str) -> bool:
    """Checks if the API key is valid."""
    # Allow master key from env for dev purposes
    master_key = os.environ.get("MATGRAPH_API_KEY")
    if master_key and api_key == master_key:
        return True
        
    keys = load_keys()
    key_info = keys.get(api_key)
    if key_info and key_info.get("active", False):
        return True
        
    return False
