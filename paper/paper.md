---
title: 'MatGraph: A Deep Learning Toolkit for Materials Property Prediction, Discovery, and Screening'
tags:
  - Python
  - materials science
  - materials informatics
  - machine learning
  - graph neural networks
  - crystallography
authors:
  - name: Himan D
    affiliation: 1
    email: himan@trinetralabs.ai
    # TODO: add ORCID before submitting, e.g. orcid: 0000-0000-0000-0000
affiliations:
  - name: Trinetra Labs
    index: 1
date: 14 November 2026
bibliography: paper.bib
---

# Summary

Computational materials scientists routinely assemble the same pipeline from
scratch: fetch crystal structures from the Materials Project [@mp2013],
featurize them, run a graph neural network potential such as M3GNet [@m3gnet2022],
MEGNet [@megnet2019], CGCNN [@cgcnn2018], or CHGNet [@chgnet2023], and then
serve or share the results. Each step means new boilerplate, new environment
conflicts, and results that are hard to reproduce.

MatGraph collapses that workflow into a single installable package with three
interfaces over one core: a command-line tool for interactive screening, a
Python SDK for notebooks and pipelines, and a GraphQL/REST API for serving
predictions to web applications and AI agents. A `pip install matgraph-cli`
gives a working CLI in seconds — including a zero-configuration demo mode —
while optional extras (`pip install "matgraph-cli[ml]"`) add the full deep
learning stack.

# Statement of need

Existing open-source building blocks (pymatgen [@pymatgen2011], MatGL
[@matgl2023], the `mp-api` client) are libraries, not workflows: a researcher
studying battery cathodes still writes custom code to query structures, batch
inference, cache repeated queries, track experiments, and export datasets.
MatGraph is the workflow layer on top of these libraries. It is designed for:

- **Screening studies** — batch property prediction with deterministic seeds,
  a reproducible SQLite cache (keyed by structure hash plus model version),
  and every result carrying machine-readable provenance (model checkpoint,
  package versions, git SHA, device, seed).
- **Discovery triage** — ML-guided elemental substitution and a genetic
  algorithm that rank hypothetical compounds by predicted stability before
  expensive density functional theory (DFT) validation.
- **Downstream integration** — dataset export (CSV/JSON/Parquet), a model
  registry, W&B-compatible experiment tracking, and an authenticated
  GraphQL/REST API with an agent plugin manifest, so predictions can flow
  into web apps, HPC pipelines, and LLM agents.

# Features

- **Six foundation potentials** through a pluggable registry: M3GNet, MEGNet,
  CGCNN, CHGNet, and OMat24/EquiformerV2 [@omat2024] stubs, predicting
  formation energy, band gap (where a head exists), forces, and stresses.
- **Seven research verticals** combining ML with scientific libraries:
  batteries, catalysis, photovoltaics, thermoelectrics, 2D materials,
  alloys/high-entropy alloys, and defects.
- **X-ray diffraction simulation**, phonon DOS, band structures, elastic,
  dielectric, and magnetic property fetch, plus an ML-to-DFT bridge that
  writes VASP and Quantum ESPRESSO inputs from ML-relaxed structures.
- **Reproducibility by default**: content-hashed caching, deterministic
  seeding, honest uncertainty (ensemble spread), and no silent fallbacks —
  unsupported model/property combinations fail with explicit errors.
- **Slim packaging**: the base install is ~30 MB and dependency-light; heavy
  stacks (PyTorch, MatGL, Gradio, PennyLane) are opt-in extras with
  actionable error messages naming the exact `pip install` command.

# Acknowledgements

MatGraph builds on the Materials Project [@mp2013], pymatgen [@pymatgen2011],
MatGL [@matgl2023], and the M3GNet [@m3gnet2022], MEGNet [@megnet2019], CGCNN
[@cgcnn2018], CHGNet [@chgnet2023], and OMat24 [@omat2024] model teams, whose
open data and open weights make this toolkit possible.

# References
