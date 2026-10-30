"""PyTorch-native crystal graph learning for materials.

    from matgraph.nn import build_crystal_graph, CrystalDataset, collate_crystals
    from matgraph.nn import CrystalGNN, Trainer

    graphs = [build_crystal_graph(lat, frac, Z, cutoff=5.0, target=[e_form]) for ...]
    loader = DataLoader(CrystalDataset(graphs), batch_size=32, shuffle=True,
                        collate_fn=collate_crystals)
    trainer = Trainer(CrystalGNN(num_targets=1), lr=1e-3)
    trainer.fit(loader, epochs=50)
    trainer.save("runs/eform.pt")
"""
from matgraph.nn.graph import (
    CrystalBatch,
    CrystalDataset,
    CrystalGraph,
    build_crystal_graph,
    collate_crystals,
    from_pymatgen,
)
from matgraph.nn.layers import CGConv, GaussianExpansion
from matgraph.nn.models import CrystalGNN
from matgraph.nn.train import Trainer, load_checkpoint

__all__ = [
    "CrystalBatch",
    "CrystalDataset",
    "CrystalGraph",
    "build_crystal_graph",
    "collate_crystals",
    "from_pymatgen",
    "CGConv",
    "GaussianExpansion",
    "CrystalGNN",
    "Trainer",
    "load_checkpoint",
]
