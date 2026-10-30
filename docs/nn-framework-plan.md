# matgraph.nn — a PyTorch-native framework for materials

Goal: the everyday materials-ML workflow (structure → graph → model → train → checkpoint)
expressed as ordinary `torch.nn.Module` code, with no DGL or MatGL dependency and no simulated numbers.

## Phase 1 (implemented)

| Piece | File | What it does |
|---|---|---|
| Graph construction | `matgraph/nn/graph.py` | Periodic neighbor graph from lattice + fractional coords (numpy). Images searched far enough for any cell shape. `max_neighbors` nearest kept. |
| Batching | `matgraph/nn/graph.py` | `collate_crystals` merges crystals with per-atom crystal ids and offset edges. `CrystalDataset` for `DataLoader`. |
| Layers | `matgraph/nn/layers.py` | `GaussianExpansion` (distance basis), `CGConv` (gated crystal convolution, LayerNorm). |
| Model | `matgraph/nn/models.py` | `CrystalGNN`: embedding → CGConv stack → mean pool → MLP. Config-driven, rebuildable. |
| Trainer | `matgraph/nn/train.py` | `fit` (optional early stopping that restores best weights), `evaluate`, `predict`, `save`, `load_checkpoint`. Reports only metrics computed from the data passed in. |
| Tests | `tests/test_nn.py` | 26 tests: periodic images (incl. non-orthogonal fcc), wrap invariance, batch offsets, batch independence, checkpoint round trip, loss decreasing on a learnable target. |

Usage:

```python
from torch.utils.data import DataLoader
from matgraph.nn import build_crystal_graph, CrystalDataset, collate_crystals, CrystalGNN, Trainer

graphs = [build_crystal_graph(lat, frac, Z, cutoff=5.0, target=[e_form]) for lat, frac, Z, e_form in data]
loader = DataLoader(CrystalDataset(graphs), batch_size=32, shuffle=True, collate_fn=collate_crystals)
trainer = Trainer(CrystalGNN(num_targets=1), lr=1e-3)
trainer.fit(loader, epochs=50)
trainer.save("runs/eform.pt")
```

## Phase 2 (done)

| Item | Status | Where |
|---|---|---|
| Fake finetune removed | Done. `simulate_finetune` and the `finetune-simulate` CLI command are deleted. `finetune` trains for real, and the active-learning retrain loop calls it. Failures are logged, not swallowed. | `matgraph/training/finetune.py`, `matgraph/core.py` |
| Pymatgen adapter tests | Done. pymatgen is installed via `uv sync --extra dev`. Tests cover `from_pymatgen` vs the plain-dict path and `as_dict()` JSONL. | `tests/test_nn_pipeline.py` |
| Data from repo sources | Done. CSV (`structure` JSON or `cif` column), JSONL of `MaterialRecord`s (pymatgen `as_dict` layout), and `graphs_from_records`. The active-learning dataset file loads directly. | `matgraph/nn/data.py` |
| MatGL pretrained bridge | **Not done, by design.** MatGL's M3GNet/CHGNet/MEGNet are different architectures (DGL/PyG, three-body terms). Their weights cannot be mapped onto `CrystalGNN`. `finetune` rejects them with an explicit error. Transfer learning works from matgraph checkpoints only. A real bridge would mean implementing those architectures natively. | `matgraph/training/finetune.py::_make_base` |
| Device and performance | Done on CPU. Neighbor search is now a per-center loop (memory O(N·S), not O(N²·S)). Measured: 64-atom cell 2 ms, 216 atoms 20 ms; a training step on 32 crystals of 10 atoms 9 ms. CUDA/MPS not measured (no GPU here). | `matgraph/nn/graph.py` |
| Uncertainty | Done as deep ensembles: std across independently seeded members. MC-dropout is not implemented. `MaterialRecord.uncertainty` is not yet filled automatically. | `matgraph/nn/ensemble.py` |
| CLI | Done. `matgraph train`, `matgraph finetune` (alias), `matgraph nn predict` (several `-c` checkpoints = ensemble std). | `matgraph/cli.py` |

## Design rules

- Everything is a `torch.nn.Module` or a plain function over tensors; no hidden global state.
- No fabricated metrics. If a metric cannot be computed, the code raises.
- Graph construction has no optional dependencies. Adapters import lazily.
- Checkpoints store the architecture config, so `load_checkpoint` never needs the original code path.
