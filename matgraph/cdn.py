"""
Zero-config caching layer for MatGraph.

Uses Python's built-in sqlite3 for a fast, persistent, local cache.
No external services, no credentials, no cost.
Repeated queries return instantly without hitting the Materials Project API.

Cache location: ~/.matgraph_cache/cache.db
Default TTL: 1 hour (3600 seconds)
"""

import os
import json
import hashlib
import time
import sqlite3
from pathlib import Path
from typing import Optional, Any

CACHE_DIR = Path.home() / ".matgraph_cache"
CACHE_DB = CACHE_DIR / "cache.db"


def _init_db() -> sqlite3.Connection:
    """Initialize the SQLite cache database."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def _get_cache_key(prefix: str, **kwargs) -> str:
    """Generate a deterministic cache key from query parameters."""
    raw = json.dumps(kwargs, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{prefix}:{digest}"


def cache_get(prefix: str, ttl: int = 3600, **kwargs) -> Optional[Any]:
    """
    Retrieve cached result if it exists and is within TTL.
    Returns None on cache miss.
    """
    key = _get_cache_key(prefix, **kwargs)
    try:
        conn = _init_db()
        row = conn.execute(
            "SELECT value, created_at FROM cache WHERE key = ?", (key,)
        ).fetchone()
        conn.close()

        if row is None:
            return None

        value, created_at = row
        if (time.time() - created_at) > ttl:
            return None  # expired

        return json.loads(value)
    except Exception:
        return None


def cache_put(prefix: str, data: Any, **kwargs):
    """Write result to cache."""
    key = _get_cache_key(prefix, **kwargs)
    try:
        conn = _init_db()
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(data, default=str), time.time()),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # caching is best-effort, never break the pipeline


def cache_clear():
    """Clear the entire cache."""
    try:
        conn = _init_db()
        conn.execute("DELETE FROM cache")
        conn.commit()
        conn.close()
    except Exception:
        pass


def cache_stats() -> dict:
    """Return cache statistics."""
    try:
        conn = _init_db()
        total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        size_bytes = CACHE_DB.stat().st_size if CACHE_DB.exists() else 0
        conn.close()
        return {
            "entries": total,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "location": str(CACHE_DB),
        }
    except Exception:
        return {"entries": 0, "size_mb": 0, "location": str(CACHE_DB)}
