"""Tests for matgraph.nn — crystal graphs, layers, model and trainer (no pymatgen needed)."""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from matgraph.nn import (
    CrystalDataset,
    CrystalGNN,
    Trainer,
    build_crystal_graph,
    collate_crystals,
    load_checkpoint,
)


def cubic(a=3.0):
    return np.eye(3) * a


def fcc_primitive(a=4.0):
    return np.array([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])


def small_model():
    return CrystalGNN(atom_dim=16, num_conv=2, cutoff=5.0, num_gaussians=16, hidden_dim=16)


# --- graph construction -------------------------------------------------------

def test_simple_cubic_has_six_nearest_neighbors():
    g = build_crystal_graph(cubic(3.0), [[0, 0, 0]], [26], cutoff=3.1)
    assert g.edge_index.shape == (2, 6)
    assert torch.allclose(g.edge_distance, torch.full((6,), 3.0))


def test_non_orthogonal_fcc_primitive_has_twelve_neighbors():
    # Primitive fcc cell is non-orthogonal; nearest shell is a/sqrt(2) = 2.828 with 12 atoms.
    g = build_crystal_graph(fcc_primitive(4.0), [[0, 0, 0]], [13], cutoff=3.0)
    assert g.edge_index.shape[1] == 12
    assert torch.allclose(g.edge_distance, torch.full((12,), 4.0 / np.sqrt(2)), atol=1e-5)


def test_max_neighbors_keeps_nearest_only():
    g = build_crystal_graph(cubic(3.0), [[0, 0, 0]], [26], cutoff=4.5, max_neighbors=4)
    assert g.edge_index.shape[1] == 4
    assert torch.allclose(g.edge_distance, torch.full((4,), 3.0))


def test_graph_invariant_under_integer_translation_of_sites():
    frac = np.array([[0.1, 0.2, 0.3], [0.6, 0.5, 0.9]])
    shifted = frac + np.array([[1, -2, 0], [0, 0, 3]])
    a = build_crystal_graph(cubic(3.5), frac, [8, 14], cutoff=4.0)
    b = build_crystal_graph(cubic(3.5), shifted, [8, 14], cutoff=4.0)
    assert torch.allclose(a.edge_distance.sort().values, b.edge_distance.sort().values, atol=1e-5)
    assert a.edge_index.shape == b.edge_index.shape


def test_no_self_loops_at_zero_distance():
    g = build_crystal_graph(cubic(3.0), [[0, 0, 0]], [26], cutoff=3.1)
    assert torch.all(g.edge_distance > 1e-6)


def test_isolated_structure_gives_empty_edges():
    g = build_crystal_graph(cubic(3.0), [[0, 0, 0]], [26], cutoff=0.5)
    assert g.edge_index.shape == (2, 0)
    assert g.edge_distance.shape == (0,)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(lattice=np.eye(2), frac_coords=[[0, 0, 0]], atomic_numbers=[1]),
        dict(lattice=cubic(), frac_coords=[[0, 0]], atomic_numbers=[1]),
        dict(lattice=cubic(), frac_coords=[[0, 0, 0]], atomic_numbers=[1, 2]),
        dict(lattice=cubic(), frac_coords=[[0, 0, 0]], atomic_numbers=[0]),
        dict(lattice=cubic(), frac_coords=[[0, 0, 0]], atomic_numbers=[1], cutoff=0),
        dict(lattice=cubic(), frac_coords=[[0, 0, 0]], atomic_numbers=[1], max_neighbors=0),
    ],
)
def test_invalid_structures_raise(kwargs):
    with pytest.raises(ValueError):
        build_crystal_graph(**kwargs)


# --- batching -----------------------------------------------------------------

def test_collate_offsets_edges_per_crystal():
    g1 = build_crystal_graph(cubic(3.0), [[0, 0, 0], [0.5, 0.5, 0.5]], [11, 17], cutoff=3.1, target=[1.0])
    g2 = build_crystal_graph(cubic(3.0), [[0, 0, 0]], [8], cutoff=3.1, target=[2.0])
    batch = collate_crystals([g1, g2])
    assert batch.num_crystals == 2
    assert batch.atomic_numbers.tolist() == [11, 17, 8]
    assert batch.crystal_index.tolist() == [0, 0, 1]
    assert batch.edge_index.max().item() == 2  # atom 2 is the only atom of crystal 2
    assert batch.edge_index[:, g1.edge_index.shape[1]:].min().item() >= 2
    assert batch.target.tolist() == [[1.0], [2.0]]


def test_collate_rejects_mixed_targets():
    g1 = build_crystal_graph(cubic(), [[0, 0, 0]], [8], target=[1.0])
    g2 = build_crystal_graph(cubic(), [[0, 0, 0]], [8])
    with pytest.raises(ValueError):
        collate_crystals([g1, g2])


# --- model --------------------------------------------------------------------

def _random_graphs(n, seed=0):
    rng = np.random.default_rng(seed)
    graphs = []
    for _ in range(n):
        natoms = int(rng.integers(1, 5))
        a = float(rng.uniform(3.0, 4.5))
        frac = rng.uniform(0, 1, size=(natoms, 3))
        z = rng.integers(1, 31, size=natoms)
        target = [float(z.mean()) / 30.0]  # composition-dependent, learnable by mean pooling
        graphs.append(build_crystal_graph(cubic(a), frac, z, cutoff=5.0, max_neighbors=8, target=target))
    return graphs


def test_forward_shape_and_config_roundtrip():
    model = CrystalGNN(atom_dim=8, num_conv=1, num_gaussians=8, hidden_dim=8, num_targets=3)
    batch = collate_crystals(_random_graphs(4))
    out = model(batch)
    assert out.shape == (4, 3)
    assert CrystalGNN(**model.config).config == model.config


def test_predictions_do_not_depend_on_batch_composition():
    torch.manual_seed(0)
    model = small_model().eval()
    graphs = _random_graphs(3, seed=1)
    together = model(collate_crystals(graphs))
    alone = torch.cat([model(collate_crystals([g])) for g in graphs])
    assert torch.allclose(together, alone, atol=1e-5)


def test_forward_handles_crystal_without_edges():
    model = small_model().eval()
    lonely = build_crystal_graph(cubic(3.0), [[0, 0, 0]], [26], cutoff=0.5)
    out = model(collate_crystals([lonely]))
    assert out.shape == (1, 1) and torch.isfinite(out).all()


def test_forward_rejects_unknown_element():
    model = CrystalGNN(max_atomic_number=10, atom_dim=8, num_conv=1, num_gaussians=8, hidden_dim=8)
    g = build_crystal_graph(cubic(), [[0, 0, 0]], [26], target=[0.0])
    with pytest.raises(ValueError):
        model(collate_crystals([g]))


def test_gradients_reach_embedding():
    torch.manual_seed(0)
    model = small_model()
    loss = model(collate_crystals(_random_graphs(4))).pow(2).mean()
    loss.backward()
    assert model.embedding.weight.grad.abs().sum().item() > 0


# --- trainer ------------------------------------------------------------------

def test_training_reduces_loss_on_learnable_target(tmp_path):
    torch.manual_seed(0)
    graphs = _random_graphs(16, seed=2)
    loader = DataLoader(CrystalDataset(graphs), batch_size=8, shuffle=True, collate_fn=collate_crystals)
    trainer = Trainer(small_model(), lr=3e-3, device="cpu")
    before = trainer.evaluate(loader)["loss"]
    trainer.fit(loader, epochs=40)
    after = trainer.evaluate(loader)["loss"]
    assert after < 0.5 * before
    assert len(trainer.history) == 40
    assert {"train_loss", "train_mae"} <= set(trainer.history[0])


def test_early_stopping_restores_best_and_records_epoch():
    torch.manual_seed(0)
    graphs = _random_graphs(12, seed=3)
    loader = DataLoader(CrystalDataset(graphs), batch_size=6, collate_fn=collate_crystals)
    trainer = Trainer(small_model(), lr=1e-2, device="cpu")
    history = trainer.fit(loader, val_loader=loader, epochs=50, patience=3)
    assert len(history) <= 50
    assert all("best_epoch" in r for r in history)


def test_patience_requires_validation_loader():
    loader = DataLoader(CrystalDataset(_random_graphs(2)), batch_size=2, collate_fn=collate_crystals)
    with pytest.raises(ValueError):
        Trainer(small_model(), device="cpu").fit(loader, epochs=1, patience=2)


def test_training_requires_targets():
    graphs = [build_crystal_graph(cubic(), [[0, 0, 0]], [8])]
    loader = DataLoader(CrystalDataset(graphs), batch_size=1, collate_fn=collate_crystals)
    with pytest.raises(ValueError):
        Trainer(small_model(), device="cpu").fit(loader, epochs=1)


def test_checkpoint_roundtrip_preserves_predictions(tmp_path):
    torch.manual_seed(0)
    graphs = _random_graphs(6, seed=4)
    loader = DataLoader(CrystalDataset(graphs), batch_size=3, collate_fn=collate_crystals)
    trainer = Trainer(small_model(), lr=1e-3, device="cpu")
    trainer.fit(loader, epochs=2)
    path = trainer.save(tmp_path / "model.pt")

    restored = load_checkpoint(path, device="cpu").eval()
    assert restored.config == trainer.model.config
    assert len(restored.history) == 2
    assert torch.allclose(trainer.predict(loader), Trainer(restored, device="cpu").predict(loader), atol=1e-6)


def test_load_rejects_foreign_file(tmp_path):
    p = tmp_path / "other.pt"
    torch.save({"hello": 1}, p)
    with pytest.raises(ValueError):
        load_checkpoint(p, device="cpu")


def test_unknown_loss_name_rejected():
    with pytest.raises(ValueError):
        Trainer(small_model(), loss="accuracy", device="cpu")
