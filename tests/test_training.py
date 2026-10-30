import tempfile, json
from pathlib import Path

def test_registry_empty():
    import os
    os.environ["MATGRAPH_TRACKING_DIR"] = tempfile.mkdtemp()
    from matgraph.training.registry import list_models
    assert list_models() == []

def test_finetune_rejects_missing_structures_and_matgl_bases(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setenv("MATGRAPH_TRACKING_DIR", str(tmp_path / "trk"))
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "dft.csv"
    csv.write_text("formula,energy\nLi2O,-1.0\nNaCl,-2.0\n")
    from matgraph.training import finetune as ft_module
    from matgraph.training.finetune import finetune
    from matgraph.exceptions import ModelLoadError, ValidationError
    # No structure column: rejected, never fabricated
    with pytest.raises(ValidationError):
        finetune(str(csv), epochs=2, target="energy", project="ut")
    # MatGL foundation models are a different architecture: honest error
    with pytest.raises(ModelLoadError):
        finetune(str(csv), base="m3gnet", epochs=2, project="ut")
    # The simulated path is gone
    assert not hasattr(ft_module, "simulate_finetune")

def test_dataset_version(tmp_path):
    import os
    os.environ["MATGRAPH_TRACKING_DIR"] = str(tmp_path / "trk2")
    f = tmp_path / "data.csv"
    f.write_text("a,b\n1,2\n")
    from matgraph.data.versioning import version_dataset, list_datasets
    vid = version_dataset(str(f))
    assert vid.startswith("v")
    assert len(list_datasets()) >= 1
