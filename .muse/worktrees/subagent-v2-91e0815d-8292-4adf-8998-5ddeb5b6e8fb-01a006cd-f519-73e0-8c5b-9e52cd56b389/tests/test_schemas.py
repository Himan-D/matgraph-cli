from matgraph.schemas import PredictRequest
import pytest

def test_valid_formula():
    r = PredictRequest(formula="LiFePO4")
    assert r.formula == "LiFePO4"

def test_invalid_formula():
    with pytest.raises(Exception):
        PredictRequest(formula="not_a_formula!!!")

def test_gap_bounds_env(monkeypatch):
    # default max 10
    with pytest.raises(Exception):
        PredictRequest(formula="Si", min_gap=20)

def test_crystal_system_normalization():
    r = PredictRequest(formula="Si", crystal_system="cubic")
    assert r.crystal_system == "Cubic"
