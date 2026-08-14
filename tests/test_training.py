import tempfile, json
from pathlib import Path

def test_registry_empty():
    import os
    os.environ["MATGRAPH_TRACKING_DIR"] = tempfile.mkdtemp()
    from matgraph.training.registry import list_models
    assert list_models() == []

def test_finetune(tmp_path):
    import os
    os.environ["MATGRAPH_TRACKING_DIR"] = str(tmp_path / "trk")
    csv = tmp_path / "dft.csv"
    csv.write_text("formula,energy\nLi2O,-1.0\nNaCl,-2.0\n")
    from matgraph.training.finetune import finetune
    res = finetune(str(csv), base="m3gnet", epochs=2, project="ut")
    assert "model_id" in res and res["n"] >= 2
    from matgraph.training.registry import list_models
    assert len(list_models()) >= 1

def test_dataset_version(tmp_path):
    import os
    os.environ["MATGRAPH_TRACKING_DIR"] = str(tmp_path / "trk2")
    f = tmp_path / "data.csv"
    f.write_text("a,b\n1,2\n")
    from matgraph.data.versioning import version_dataset, list_datasets
    vid = version_dataset(str(f))
    assert vid.startswith("v")
    assert len(list_datasets()) >= 1
