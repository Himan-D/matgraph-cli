"""
Zero-config caching layer for MatGraph.
Now using standard `diskcache`.
"""
import json
import hashlib
from typing import Optional, Any
import diskcache

def _cache():
    from matgraph.settings import settings
    return diskcache.Cache(str(settings.cache_dir))

def _get_cache_key(prefix: str, **kwargs) -> str:
    raw = json.dumps(kwargs, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{prefix}:{digest}"

def cache_get(prefix: str, ttl: Optional[int] = None, **kwargs) -> Optional[Any]:
    cache = _cache()
    key = _get_cache_key(prefix, **kwargs)
    return cache.get(key)

def cache_put(prefix: str, data: Any, **kwargs):
    cache = _cache()
    from matgraph.settings import get_ttl
    ttl = get_ttl(prefix)
    key = _get_cache_key(prefix, **kwargs)
    cache.set(key, data, expire=ttl)

def cache_stats() -> dict:
    cache = _cache()
    return {
        "entries": len(list(cache.iterkeys())),
        "size_mb": round(cache.volume() / (1024 * 1024), 2),
        "location": cache.directory
    }

def cache_clear():
    _cache().clear()
