import os
import json
from pathlib import Path
from typing import Optional

CONFIG_DIR = Path.home() / ".matgraph"
CONFIG_FILE = CONFIG_DIR / "config.json"

def save_api_key(api_key: str):
    """Save the API key to the local config file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    config = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError:
                pass
                
    config["mp_api_key"] = api_key
    
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)
        
    # Restrict permissions so only the user can read the config file
    CONFIG_FILE.chmod(0o600)

def get_api_key() -> Optional[str]:
    """
    Retrieve the API key from the environment variable or the local config file.
    Environment variable takes precedence.
    """
    # 1. Check environment variable
    api_key = os.environ.get("MP_API_KEY")
    if api_key:
        return api_key
        
    # 2. Check config file
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            try:
                config = json.load(f)
                return config.get("mp_api_key")
            except json.JSONDecodeError:
                return None
                
    return None
