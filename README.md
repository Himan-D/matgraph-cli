<div align="center">
  <h1>MatGraph</h1>
  <p><strong>Deep Learning toolkit for Materials Science researchers.</strong></p>
  <p>Predict material properties, discover new compounds, simulate diffraction patterns, and serve predictions via API -- all from one package.</p>

  [![PyPI version](https://badge.fury.io/py/matgraph-cli.svg)](https://badge.fury.io/py/matgraph-cli)
  [![Downloads](https://static.pepy.tech/badge/matgraph-cli)](https://pepy.tech/project/matgraph-cli)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Himan-D/matgraph-cli/blob/main/notebooks/MatGraph_Tutorial.ipynb)
</div>

---

## Why MatGraph?

Researchers spend weeks writing boilerplate to fetch crystal data, engineer features, train GNNs, and serve predictions. MatGraph collapses that into a single `pip install`.

| Problem | MatGraph solution |
|---|---|
| Fetching crystal structures from Materials Project | `sdk.predict("LiFePO4")` |
| Running MatGL inference (M3GNet/MEGNet/CGCNN) | `matgraph predict LiFePO4 --model m3gnet` |
| Exploring hypothetical new materials (heuristic) | `matgraph substitute LiFePO4 Li Na` (ML-guided, not GNoME-scale) |
| Simulating XRD patterns | `matgraph xrd LiFePO4` |
| Serving predictions to a web app | Async GraphQL + REST `/v1/predict` with hashed API keys |
| Caching repeated queries | Reproducible SQLite cache (structure_hash + model_version) |

---

## Installation

```bash
# Recommended (fastest)
uv tool install matgraph-cli

# Or standard pip
pip install matgraph-cli
```

Set your free [Materials Project](https://materialsproject.org/) API key:

```bash
export MP_API_KEY="your_key_here"
```

---

## Quickstart

### CLI

```bash
# Predict band gap and formation energy
matgraph predict LiFePO4

# All three models are real now (separate checkpoints)
matgraph predict LiFePO4 --model m3gnet
matgraph predict LiFePO4 --model megnet --seed 42
matgraph predict LiFePO4 --model cgcnn

# ML-guided heuristic discovery (not GNoME-scale)
matgraph substitute LiFePO4 Li Na

# Simulate X-Ray Diffraction pattern
matgraph xrd LiFePO4

# Evaluate formation-energy MAE (band_gap unavailable — no UQ model)
matgraph evaluate LiFePO4 --model m3gnet

# Filter by physical constraints
matgraph predict LiFePO4 --min-gap 1.5 --crystal-system Cubic

# Export dataset for downstream ML
matgraph predict LiFePO4 --save dataset.csv --format csv --cif

# Check version
matgraph --version
```

### Python SDK (Jupyter Notebooks, Scripts, Pipelines)

```python
from matgraph import MatGraphSDK

sdk = MatGraphSDK()

# Predict properties
results = sdk.predict("LiFePO4", model="m3gnet")
print(results[0]["m3gnet_energy"])

# Generative discovery
discovery = sdk.substitute("LiFePO4", element_out="Li", element_in="Na")
print("Stable" if discovery["is_more_stable"] else "Unstable")

# XRD simulation
xrd = sdk.xrd("LiFePO4")

# Model evaluation — band_gap MAE is None until a real band-gap model ships
metrics = sdk.evaluate("LiFePO4", model="m3gnet")
print(f"Formation energy MAE: {metrics['formation_energy_mae']}")
```

### GraphQL API

Start the server:

```bash
uvicorn matgraph.graphql_app:app --reload
```

Generate an API key:

```bash
matgraph auth generate --user "my-app"
# Output: mg_S8jvhzo58p6XQE_...
```

Query the API:

```bash
curl -X POST http://localhost:8000/graphql \
  -H "Content-Type: application/json" \
  -H "x-api-key: <YOUR_KEY>" \
  -d '{"query": "{ predictMaterial(formula: \"LiFePO4\") { predictedFormEnergy } }"}'
```

Or open `http://localhost:8000/graphql` for the interactive GraphiQL playground.

---

## Features

### Deep Learning Models (2.1 — all three are real)

| Model | Predicts | Architecture | Checkpoint | Band gap |
|---|---|---|---|---|
| **M3GNet** | Energy, Forces, Stresses, Formation energy | Multi-body universal potential | `M3GNet-PES-MatPES-PBE-2025.2` + `M3GNet-Eform-MP-2019.4.1` | `None` (no head, use `true_band_gap`) |
| **MEGNet** | Formation energy + Band gap | MatErials Graph Network | `MEGNet-MP-2019.4.1-Eform` + `MEGNet-MP-2019.4.1-BandGap-mfi` | ✅ ML head |
| **CGCNN** | Formation energy + Band gap (via MEGNet bandgap head as proxy) | Crystal Graph CNN | `MEGNet-MP-2019.4.1-BandGap-mfi` proxy | ✅ ML head |

> **2.1 fix:** `cgcnn/megnet` are no longer aliases to `m3gnet` — each has its own checkpoint. M3GNet still has no band-gap head (`predicted_band_gap=None`).

### ML-guided heuristic discovery (experimental)

Heuristic elemental substitution + simple GA ranking via M3GNet energies. Useful for triage, **not** GNoME-scale generative discovery.

```bash
matgraph substitute LiFePO4 Li Na
# Predicts: NaFePO4 stability vs LiFePO4 (heuristic, validate with DFT)
```

### XRD Simulation

Generate theoretical Cu-Ka X-Ray Diffraction patterns for any material. Useful for matching experimental peaks against predicted structures.

```bash
matgraph xrd LiFePO4
```

### Reproducible cache (2.0)

SQLite + WAL at `~/.matgraph_cache/cache.db` (override `MATGRAPH_CACHE_DIR`), key = `material_id+structure_hash+model+checkpoint+code_version+params`. Reproducibility via `provenance` field on every prediction.

```bash
matgraph cache stats    # View cache size and entry count
matgraph cache clear    # Wipe the cache
```

### Hashed API keys (2.0)

Keys are `mg_*`, stored as `sha256` with `scopes/expiry/revocation` in `~/.matgraph_keys.json` (override `MATGRAPH_AUTH_KEYS_FILE`). `MATGRAPH_API_KEY` master key still supported. Not multi-tenant authz — local research use.

```bash
matgraph auth generate --user "research-team-A"
```

### Dataset Export

Export predictions to CSV or JSON for use in pandas, scikit-learn, or any ML pipeline. Optionally export 3D crystal structures as `.cif` files.

```bash
matgraph predict LiFePO4 --save results.json --format json --cif
```

---

## Architecture

```
matgraph/
  __init__.py
  sdk.py           # SDK (predict/substitute/xrd/... + DataFrame)
  cli.py           # Typer CLI
  core.py          # Orchestration shim (re-exports data/models/...)
  client.py        # Materials Project client
  models.py        # M3GNet registry (settings.pes_model)
  schemas.py       # Pydantic validation, no hardcodes
  settings.py      # Central MATGRAPH_* settings
  cdn.py           # WAL SQLite cache
  auth.py          # sha256 keys + scopes/expiry
  ga.py            # Heuristic GA (param-driven)
  graphql_app.py   # GraphQL + REST /v1/predict + /health
  data/            # (v2 split) materials_project
  simulation/      # xrd/phonon/relax
  dft/             # vasp/qe input generation
  properties/      # stability/elastic/...
```

### Tech Stack

| Layer | Technology |
|---|---|
| ML | PyTorch, scikit-learn |
| Data | pymatgen, mp-api (Materials Project) |
| API | FastAPI, Strawberry GraphQL |
| CLI | Typer, Rich |
| Cache | SQLite3 (stdlib) |
| Build | uv, Hatchling |

---

## Changelog

### v1.1.0
- Replaced AWS CDN with zero-config SQLite cache
- Added `matgraph cache stats` and `matgraph cache clear` commands

### v1.0.0
- Added AWS S3/CloudFront CDN caching layer

### v0.9.0
- Added API key generation system (`matgraph auth generate`)
- Multi-tenant key validation for GraphQL server

### v0.8.0
- Added API key security to GraphQL Engine

### v0.7.0
- Added Python SDK (`MatGraphSDK`) for Jupyter Notebooks and scripts

### v0.6.0
- GNoME-inspired generative discovery (`matgraph substitute`)
- X-Ray Diffraction simulation (`matgraph xrd`)
- M3GNet universal potential architecture
- Model evaluation with MAE (`matgraph evaluate`)
- CIF structure export (`--cif` flag)

### v0.5.0
- MEGNet architecture
- Multi-property predictions (band gap + formation energy)
- Advanced CLI filtering and dataset export

### v0.1.0
- Initial release with CGCNN, Materials Project integration, GraphQL API, and CLI

---

## Contributing

```bash
git clone https://github.com/Himan-D/matgraph-cli.git
cd matgraph-cli
uv sync
uv run pytest
```

Open an issue before submitting major pull requests.

---

## Citation

If you use MatGraph in your research, please cite:

```bibtex
@software{matgraph2025,
  author = {Himan},
  title = {MatGraph: Deep Learning Toolkit for Materials Science},
  url = {https://github.com/Himan-D/matgraph-cli},
  year = {2025}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
  <p>Built by <a href="https://github.com/Himan-D">Himan</a> at <a href="https://trinetralabs.ai">Trinetra Labs</a></p>
</div>
