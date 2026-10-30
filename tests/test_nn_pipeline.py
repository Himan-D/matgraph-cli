"""End-to-end tests for matgraph.nn data loading, finetune, ensembles and CLI."""
import json
import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from matgraph.exceptions import ModelLoadError, ValidationError
from matgraph.nn import CrystalDataset, Trainer, build_crystal_graph, collate_crystals, load_checkpoint
from matgraph.nn.data import load_dataset, train_val_split
from matgraph.nn.ensemble import predict_ensemble, train_ensemble
from matgraph.nn.graph import graph_from_dict, structure_to_dict
from matgraph.nn.models import CrystalGNN

SMALL = dict(atom_dim=12, num_conv=1, num_gaussians=12, hidden_dim=12, cutoff=5.0)


def _structures(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        natoms = int(rng.integers(1, 4))
        a = float(rng.uniform(3.0, 4.5))
        out.append(
            {
                "lattice": (np.eye(3) * a).tolist(),
                "frac_coords": rng.uniform(0, 1, size=(natoms, 3)).tolist(),
                "atomic_numbers": rng.integers(1, 31, size=natoms).tolist(),
            }
        )
    return out


def _target(structure):
    return float(np.mean(structure["atomic_numbers"])) / 30.0


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    from matgraph.settings import settings

    monkeypatch.setenv("MATGRAPH_TRACKING_DIR", str(tmp_path / "trk"))
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setattr(settings, "cache_dir", tmp_path / "cache")  # keep checkpoints out of ~/.matgraph_cache
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write_csv(path, n=12, seed=0):
    lines = ["structure,formation_energy_per_atom"]
    for s in _structures(n, seed):
        lines.append(f"\"{json.dumps(s).replace(chr(34), chr(34) * 2)}\",{_target(s):.6f}")
    path.write_text("\n".join(lines) + "\n")
    return path


# --- loading ------------------------------------------------------------------

def test_csv_loads_json_structures_with_targets(tmp_path):
    p = _write_csv(tmp_path / "data.csv", n=6)
    ds = load_dataset(p, "formation_energy_per_atom", cutoff=5.0)
    assert len(ds) == 6
    assert all(g.target is not None and g.target.shape == (1,) for g in ds.graphs)


def test_csv_rejects_non_numeric_target(tmp_path):
    p = tmp_path / "bad.csv"
    s = json.dumps(_structures(1)[0]).replace('"', '""')
    p.write_text(f'structure,y\n"{s}",oops\n')
    with pytest.raises(ValidationError, match="not a number"):
        load_dataset(p, "y")


def test_csv_rejects_missing_target_column(tmp_path):
    p = _write_csv(tmp_path / "data.csv", n=2)
    with pytest.raises(ValidationError, match="not in"):
        load_dataset(p, "band_gap")


def test_unsupported_file_type_rejected(tmp_path):
    p = tmp_path / "data.xlsx"
    p.write_text("x")
    with pytest.raises(ValidationError):
        load_dataset(p, "y")


def test_train_val_split_is_seeded_and_partitions():
    graphs = [build_crystal_graph(np.eye(3) * 3, [[0, 0, 0]], [8], target=[float(i)]) for i in range(10)]
    tr1, va1 = train_val_split(graphs, val_fraction=0.3, seed=1)
    tr2, va2 = train_val_split(graphs, val_fraction=0.3, seed=1)
    assert [float(g.target[0]) for g in va1] == [float(g.target[0]) for g in va2]
    assert len(tr1) + len(va1) == 10 and len(va1) == 3
    with pytest.raises(ValueError):
        train_val_split(graphs[:4])


def test_graph_from_dict_matches_plain_structure():
    s = _structures(1, seed=5)[0]
    g = graph_from_dict(s, cutoff=5.0)
    assert g.atomic_numbers.tolist() == s["atomic_numbers"]
    with pytest.raises(ValueError):
        graph_from_dict({"lattice": s["lattice"]}, cutoff=5.0)


# --- pymatgen interop (runs when pymatgen is installed) -----------------------

def test_pymatgen_structure_matches_plain_dict_and_jsonl(tmp_path):
    pymatgen = pytest.importorskip("pymatgen.core")
    from matgraph.nn import from_pymatgen

    s = pymatgen.Structure(pymatgen.Lattice.cubic(3.0), ["Li", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    d = structure_to_dict(s)
    assert sorted(d["atomic_numbers"]) == sorted([3, 17])
    g_direct = from_pymatgen(s, cutoff=4.0)
    g_dict = graph_from_dict(d, cutoff=4.0)
    assert torch.allclose(g_direct.edge_distance.sort().values, g_dict.edge_distance.sort().values)

    # MaterialRecord-style JSONL uses pymatgen's as_dict() layout
    rows = [{"structure": s.as_dict(), "formation_energy_per_atom": -1.5} for _ in range(6)]
    p = tmp_path / "al.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ds = load_dataset(p, "formation_energy_per_atom", cutoff=4.0)
    assert len(ds) == 6
    assert torch.allclose(ds.graphs[0].edge_distance.sort().values, g_dict.edge_distance.sort().values)


def test_jsonl_without_target_rejected(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"structure": _structures(1)[0]}) + "\n")
    with pytest.raises(ValidationError, match="no 'formation_energy_per_atom'"):
        load_dataset(p, "formation_energy_per_atom")


# --- finetune -----------------------------------------------------------------

def test_finetune_trains_registers_and_checkpoint_reloads(tracking):
    from matgraph.training.finetune import finetune
    from matgraph.training.registry import list_models

    data = _write_csv(tracking / "data.csv", n=12)
    res = finetune(str(data), epochs=3, project="ut-real", patience=None, out_dir=str(tracking / "ck"),
                   batch_size=4, cutoff=5.0)
    m = res["metrics"]
    assert m["n_train"] + m["n_val"] == 12
    assert math.isfinite(m["val_mae"]) and m["val_mae"] >= 0
    assert m["epochs_run"] == 3 and 1 <= m["best_epoch"] <= 3
    assert len(res["history"]) == 3

    reloaded = load_checkpoint(res["checkpoint"], device="cpu")
    assert isinstance(reloaded, CrystalGNN)
    assert any(r["id"] == res["model_id"] for r in list_models())


def test_finetune_continues_from_matgraph_checkpoint(tracking):
    from matgraph.training.finetune import finetune

    data = _write_csv(tracking / "data.csv", n=10)
    first = finetune(str(data), epochs=1, project="ut-a", out_dir=str(tracking / "ck"), cutoff=5.0)
    second = finetune(str(data), base=first["checkpoint"], epochs=1, project="ut-b",
                      out_dir=str(tracking / "ck"), cutoff=5.0)
    assert second["model_id"] != first["model_id"]
    assert second["metrics"]["n_val"] == first["metrics"]["n_val"]


def test_finetune_unknown_base_rejected(tracking):
    from matgraph.training.finetune import finetune
    from matgraph.exceptions import ValidationError as VE

    data = _write_csv(tracking / "data.csv", n=6)
    with pytest.raises(VE):
        finetune(str(data), base="does-not-exist.pt", epochs=1)
    with pytest.raises(ModelLoadError):
        finetune(str(data), base="chgnet", epochs=1)


# --- ensembles ----------------------------------------------------------------

def _loaders(n=8, seed=7):
    graphs = [graph_from_dict(s, cutoff=5.0, target=[_target(s)]) for s in _structures(n, seed)]
    return DataLoader(CrystalDataset(graphs), batch_size=4, collate_fn=collate_crystals), graphs


def test_ensemble_members_disagree_and_std_is_reported():
    loader, graphs = _loaders()
    members = train_ensemble(loader, None, n_members=3, epochs=1,
                             make_model=lambda: CrystalGNN(**SMALL), seed=0)
    mean, std = predict_ensemble(members, loader)
    assert mean.shape == (len(graphs), 1) and std.shape == (len(graphs), 1)
    assert (std > 0).any()


def test_ensemble_refuses_single_member():
    loader, _ = _loaders(n=4)
    with pytest.raises(ValueError):
        train_ensemble(loader, None, n_members=1, epochs=1)
    with pytest.raises(ValueError):
        predict_ensemble([CrystalGNN(**SMALL)], loader)


# --- CLI ----------------------------------------------------------------------

def test_cli_train_and_predict(tracking):
    from typer.testing import CliRunner
    from matgraph.cli import app

    runner = CliRunner()
    data = _write_csv(tracking / "data.csv", n=10)
    res = runner.invoke(app, ["train", "--data", str(data), "--epochs", "2", "--patience", "0",
                              "--project", "cli-ut"])
    assert res.exit_code == 0, res.output
    assert "Validation MAE" in res.output

    ckpts = sorted((tracking / "cache" / "models").glob("cli-ut-*.pt"))
    assert len(ckpts) == 1, "train should write exactly one checkpoint into the cache dir"
    out_csv = tracking / "pred.csv"
    res = runner.invoke(app, ["nn", "predict", "--data", str(data), "-c", str(ckpts[0]),
                              "--out", str(out_csv)])
    assert res.exit_code == 0, res.output
    assert out_csv.read_text().splitlines()[0] == "index,prediction,target"


def test_cli_predict_ensemble_shows_std(tracking):
    from typer.testing import CliRunner
    from matgraph.cli import app
    from matgraph.nn import Trainer as T

    data = _write_csv(tracking / "data.csv", n=6)
    paths = []
    for k in range(2):
        torch.manual_seed(k)
        m = CrystalGNN(**SMALL)
        path = tracking / f"m{k}.pt"
        T(m, device="cpu").save(path)
        paths.append(str(path))
    res = CliRunner().invoke(app, ["nn", "predict", "--data", str(data), "-c", paths[0], "-c", paths[1]])
    assert res.exit_code == 0, res.output
    assert "std" in res.output
