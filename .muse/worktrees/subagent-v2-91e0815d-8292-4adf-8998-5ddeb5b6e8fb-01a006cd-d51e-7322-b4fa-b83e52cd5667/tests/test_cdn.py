import os, tempfile, pathlib
from matgraph import cdn

def test_cache_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("MATGRAPH_CACHE_DIR", str(tmp_path))
    from importlib import reload
    import matgraph.settings as st
    reload(st)
    reload(cdn)
    cdn.cache_put("t", {"x": 1}, k="a")
    assert cdn.cache_get("t", k="a") == {"x": 1}
    stats = cdn.cache_stats()
    assert stats["entries"] >= 1
    assert tmp_path.as_posix() in stats["location"]
