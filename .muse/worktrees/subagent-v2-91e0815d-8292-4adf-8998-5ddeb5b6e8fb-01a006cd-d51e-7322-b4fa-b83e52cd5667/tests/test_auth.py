import tempfile, pathlib
from importlib import reload
import matgraph.auth as auth

def test_hashed_keys(tmp_path, monkeypatch):
    # isolated settings reload needed before import
    monkeypatch.setenv("MATGRAPH_AUTH_KEYS_FILE", str(tmp_path/"keys.json"))
    import matgraph.settings as st
    reload(st)
    reload(auth)
    k = auth.generate_api_key("alice", ttl_days=1)
    assert k.startswith("mg_")
    assert auth.is_valid_key(k)
    # stored file should not contain raw key
    text = (tmp_path/"keys.json").read_text()
    assert k not in text
    assert auth.revoke_key(k)
    assert not auth.is_valid_key(k)
