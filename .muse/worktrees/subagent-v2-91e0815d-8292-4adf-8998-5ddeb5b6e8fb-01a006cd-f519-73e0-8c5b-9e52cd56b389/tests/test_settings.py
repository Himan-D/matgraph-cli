import os
from importlib import reload
import matgraph.settings as s

def test_env_overrides(monkeypatch):
    monkeypatch.setenv("MATGRAPH_PES_MODEL", "custom")
    monkeypatch.setenv("MATGRAPH_GA_ELEMENTS", "Fe,O")
    reload(s)
    assert s.settings.pes_model == "custom"
    assert s.settings.ga_allowed_elements == ["Fe","O"]
    # cleanup
    monkeypatch.delenv("MATGRAPH_PES_MODEL", raising=False)
    monkeypatch.delenv("MATGRAPH_GA_ELEMENTS", raising=False)
    reload(s)
