# Models

MatGraph ships with three deep learning architectures for material property prediction.

## CGCNN (Crystal Graph Convolutional Neural Network)

**Paper:** [Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301) (Xie & Grossman, 2018)

**Predicts:** Band gap, Formation energy per atom

**How it works:**
1. Represents a crystal as a graph where nodes are atoms and edges are bonds
2. Applies graph convolutions to learn atom-level representations
3. Pools atom features into a crystal-level descriptor
4. Passes through fully connected layers for property prediction

**Usage:**

```bash
matgraph predict LiFePO4 --model cgcnn
```

```python
results = sdk.predict("LiFePO4", model="cgcnn")
```

---

## MEGNet (MatErials Graph Network)

**Paper:** [Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals](https://pubs.acs.org/doi/10.1021/acs.chemmater.9b01294) (Chen et al., 2019)

**Predicts:** Band gap, Formation energy per atom

**How it works:**
1. Encodes atoms, bonds, and global state as separate feature vectors
2. Updates all three through message-passing layers
3. Global state captures crystal-wide properties
4. Outputs property predictions from the global state vector

**Key difference from CGCNN:** MEGNet explicitly models global state information (total charge, composition features), which improves accuracy for properties that depend on the overall crystal rather than individual atoms.

**Usage:**

```bash
matgraph predict LiFePO4 --model megnet
```

```python
results = sdk.predict("LiFePO4", model="megnet")
```

---

## M3GNet (Multi-body M3GNet)

**Paper:** [A universal graph deep learning interatomic potential for the periodic table](https://www.nature.com/articles/s43588-022-00349-3) (Chen & Ong, 2022)

**Predicts:** Total energy, Atomic forces, Stress tensor

**How it works:**
1. Encodes atoms with embeddings based on atomic number
2. Computes 2-body interactions (pairwise distances)
3. Computes 3-body interactions (bond angles)
4. Multi-layer message passing with both 2-body and 3-body information
5. Predicts energy as the sum of atomic contributions
6. Forces and stresses are derived from energy gradients

**Key difference from CGCNN/MEGNet:** M3GNet is a universal interatomic potential. Instead of predicting a single property, it predicts the potential energy surface, from which forces and stresses can be computed. This makes it suitable for molecular dynamics and structure relaxation.

**Usage:**

```bash
matgraph predict LiFePO4 --model m3gnet
```

```python
results = sdk.predict("LiFePO4", model="m3gnet")
print(f"Energy: {results[0]['m3gnet_energy']} eV")
print(f"Forces: {results[0]['m3gnet_forces']}")
```

---

## Model Comparison

| Property | CGCNN | MEGNet | M3GNet |
|---|---|---|---|
| Band gap | Yes | Yes | No |
| Formation energy | Yes | Yes | No |
| Total energy | No | No | Yes |
| Atomic forces | No | No | Yes |
| Stress tensor | No | No | Yes |
| Architecture | GCN | Graph Network | Multi-body GNN |
| Best for | Property screening | Property screening | Stability analysis |
