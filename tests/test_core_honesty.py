from unittest.mock import MagicMock, patch
from matgraph.core import run_pipeline, _provenance
from matgraph.settings import settings

def _fake_docs():
    m = MagicMock()
    m.material_id = "mp-123"
    m.formula_pretty = "Si"
    m.band_gap = 1.1
    m.formation_energy_per_atom = -0.5
    m.density = 2.3
    m.symmetry.crystal_system.name = "Cubic"
    # minimal structure mock
    struct = MagicMock()
    struct.composition.elements = [MagicMock()]
    struct.composition.weight = 28
    struct.composition.num_atoms = 1
    struct.volume = 20
    struct.density = 2.3
    m.structure = struct
    return [m]

import numpy as np
@patch("matgraph.core.fetch_materials_data", return_value=_fake_docs())
@patch("matgraph.core.m3gnet_predict_pes", return_value=(1.0, np.array([[0,0,0]], dtype=float), np.array([0]*6, dtype=float)))
@patch("matgraph.models.M3GNetPotential.predict_band_gap", return_value=None)
@patch("matgraph.models.M3GNetPotential.predict_eform", return_value=-0.4)
def test_band_gap_is_none(mock_eform, mock_bg, *_):
    res = run_pipeline("Si", api_key="dummy", seed=42)
    assert res[0]["predicted_band_gap"] is None
    assert res[0]["band_gap_source"] == "mp_experimental"
    assert "provenance" in res[0]
    assert res[0]["provenance"]["seed"] == 42
    assert res[0]["provenance"]["m3gnet_pes_model"] == settings.pes_model

def test_provenance_uses_settings():
    p = _provenance(seed=7)
    assert p["seed"] == 7
    assert p["m3gnet_pes_model"] == settings.pes_model
