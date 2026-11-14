"""Demo mode: zero-config sample output. Must pass on the SLIM install (no MP key,
no torch/pymatgen/mp-api). Run in CI's slim job as well as the full suite."""

import pytest
from typer.testing import CliRunner

from matgraph.cli import app
from matgraph.demo import (
    DEMO_FORMULAS,
    demo_battery,
    demo_predict,
    demo_substitute,
    demo_xrd,
)
from matgraph.exceptions import DataNotFoundError, ValidationError

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_key(monkeypatch, tmp_path):
    # Prove demo works with no credentials anywhere. Settings are read at
    # import time, so reload like the existing test_cdn/test_settings tests.
    from importlib import reload

    import matgraph.settings as st

    monkeypatch.delenv("MP_API_KEY", raising=False)
    monkeypatch.setenv("MATGRAPH_CONFIG_FILE", str(tmp_path / "config.json"))
    monkeypatch.setenv("MATGRAPH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MATGRAPH_AUTH_KEYS_FILE", str(tmp_path / "keys.json"))
    reload(st)
    yield
    monkeypatch.delenv("MATGRAPH_CONFIG_FILE", raising=False)
    monkeypatch.delenv("MATGRAPH_CACHE_DIR", raising=False)
    monkeypatch.delenv("MATGRAPH_AUTH_KEYS_FILE", raising=False)
    reload(st)


def test_demo_predict_schema_matches_pipeline():
    for formula in DEMO_FORMULAS:
        rows = demo_predict(formula, model="m3gnet")
        assert len(rows) == 1
        r = rows[0]
        for key in (
            "material_id",
            "formula",
            "true_band_gap",
            "predicted_band_gap",
            "band_gap_source",
            "true_form_energy",
            "predicted_form_energy",
            "m3gnet_energy",
            "m3gnet_forces",
            "m3gnet_stresses",
            "crystal_system",
            "features",
            "model_used",
            "provenance",
        ):
            assert key in r, f"{formula} missing {key}"
        assert r["provenance"]["mode"] == "demo"
        assert r["model_used"] == "M3GNET-DEMO"
        assert r["features"]["density"] > 0
        assert r["features"]["volume"] > 0


def test_demo_predict_gap_head_models_get_sample_gap():
    r = demo_predict("Si", model="megnet")[0]
    assert r["predicted_band_gap"] is not None
    assert r["band_gap_source"] == "demo_sample_gap_head"
    assert demo_predict("Si", model="m3gnet")[0]["predicted_band_gap"] is None


def test_demo_predict_unknown_formula():
    with pytest.raises(DataNotFoundError, match="Demo formulas"):
        demo_predict("Fe2O3")


def test_demo_predict_case_insensitive():
    assert demo_predict("lifepo4")[0]["formula"] == "LiFePO4"


def test_demo_xrd_is_real_crystallography():
    xrd = demo_xrd("Si")
    assert len(xrd["two_theta"]) == len(xrd["intensity"]) == len(xrd["hkls"])
    # Diamond-cubic Si 111 reflection at 28.44 deg (Cu-Ka) — must be exact.
    peaks = sorted(zip(xrd["two_theta"], xrd["intensity"], xrd["hkls"]), key=lambda p: -p[1])
    assert abs(peaks[0][0] - 28.44) < 0.1
    assert peaks[0][1] == 100.0
    assert [1, 1, 1] in peaks[0][2]
    assert xrd["provenance"]["mode"] == "demo"
    assert "xrd" in xrd["provenance"]["computed"]


def test_demo_substitute():
    res = demo_substitute("LiFePO4", "Li", "Na")
    assert res["original"]["formula"] == "LiFePO4"
    assert res["hypothetical"]["formula"] == "NaFePO4"
    assert res["hypothetical"]["energy"] != res["original"]["energy"]
    assert res["provenance"]["mode"] == "demo"


def test_demo_substitute_validation():
    with pytest.raises(ValidationError, match="must differ"):
        demo_substitute("Si", "Si", "Si")
    with pytest.raises(ValidationError, match="not found"):
        demo_substitute("Si", "Li", "Na")
    with pytest.raises(ValidationError, match="Invalid formula"):
        demo_substitute("not-a-formula!!", "Li", "Na")


def test_sdk_demo_mode_no_key():
    from matgraph import MatGraphSDK

    sdk = MatGraphSDK(demo=True)
    rows = sdk.predict("Si")
    assert rows[0]["provenance"]["mode"] == "demo"
    assert sdk.xrd("Si")["formula"] == "Si"
    assert sdk.substitute("LiFePO4", "Li", "Na")["hypothetical"]["formula"] == "NaFePO4"


def test_demo_battery_capacity_is_exact():
    m = demo_battery("LiFePO4")
    # Theoretical LiFePO4 capacity: 1*96485/(3.6*157.76) = 169.9 mAh/g
    assert abs(m["theoretical_capacity_mah_g"] - 169.9) < 0.05
    assert m["carrier"] == "Li"
    assert m["provenance"]["mode"] == "demo"
    assert "battery.capacity" in m["provenance"]["computed"]
    assert "battery.voltage" in m["provenance"]["sample"]
    with pytest.raises(DataNotFoundError, match="Demo formulas"):
        demo_battery("Fe2O3")


def test_sdk_battery_demo():
    from matgraph import MatGraphSDK

    sdk = MatGraphSDK(demo=True)
    assert sdk.battery("LiFePO4")["theoretical_capacity_mah_g"] > 0


def test_sdk_still_requires_key_when_not_demo():
    from matgraph import MatGraphSDK

    # No key in env or (redirected) config -> must raise.
    with pytest.raises(ValueError, match="demo=True"):
        MatGraphSDK()


def test_cli_predict_demo():
    res = runner.invoke(app, ["predict", "Si", "--demo"])
    assert res.exit_code == 0, res.output
    assert "DEMO MODE" in res.output
    assert "Si" in res.output


def test_cli_xrd_demo():
    res = runner.invoke(app, ["xrd", "Si", "--demo"])
    assert res.exit_code == 0, res.output
    assert "28.47" in res.output


def test_cli_substitute_demo():
    res = runner.invoke(app, ["substitute", "LiFePO4", "Li", "Na", "--demo"])
    assert res.exit_code == 0, res.output
    assert "NaFePO4" in res.output


def test_cli_no_key_suggests_demo():
    res = runner.invoke(app, ["predict", "Si"])
    assert res.exit_code == 1
    assert "--demo" in res.output


def test_cli_battery_demo():
    res = runner.invoke(app, ["battery", "LiFePO4", "--demo"])
    assert res.exit_code == 0, res.output
    assert "169.89" in res.output
    assert "DEMO MODE" in res.output
