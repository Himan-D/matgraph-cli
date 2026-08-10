<div align="center">
  <h1>MatGraph</h1>
  <p><strong>Deep Learning toolkit for Materials Science researchers.</strong></p>
  <p>Predict material properties, discover new compounds, simulate diffraction patterns, and serve predictions via API -- all from one package.</p>

  [![PyPI](https://img.shields.io/pypi/v/matgraph-cli?color=blue&label=PyPI)](https://pypi.org/project/matgraph-cli/)
  [![Python](https://img.shields.io/pypi/pyversions/matgraph-cli)](https://pypi.org/project/matgraph-cli/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Downloads](https://static.pepy.tech/badge/matgraph-cli)](https://pepy.tech/project/matgraph-cli)
</div>

---

## Why MatGraph?

Researchers spend weeks writing boilerplate to fetch crystal data, engineer features, train GNNs, and serve predictions. MatGraph collapses that into a single `pip install`.

| Problem | MatGraph solution |
|---|---|
| Fetching crystal structures from Materials Project | `sdk.predict("LiFePO4")` |
| Training CGCNN / MEGNet / M3GNet from scratch | Pre-wired architectures, ready to run |
| Exploring hypothetical new materials | `matgraph substitute LiFePO4 Li Na` |
| Simulating XRD patterns | `matgraph xrd LiFePO4` |
| Serving predictions to a web app | Async GraphQL API with API key auth |
| Caching repeated queries | Built-in SQLite cache, zero config |

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

# Use a different model architecture
matgraph predict LiFePO4 --model m3gnet

# Discover new materials via elemental substitution
matgraph substitute LiFePO4 Li Na

# Simulate X-Ray Diffraction pattern
matgraph xrd LiFePO4

# Evaluate model accuracy (MAE) against ground truth
matgraph evaluate LiFePO4 --model megnet

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

# Model evaluation
metrics = sdk.evaluate("LiFePO4", model="megnet")
print(f"Band gap MAE: {metrics['band_gap_mae']}")
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

### Deep Learning Models

| Model | Predicts | Architecture |
|---|---|---|
| **CGCNN** | Band gap, Formation energy | Crystal Graph Convolutional Neural Network |
| **MEGNet** | Band gap, Formation energy | MatErials Graph Network |
| **M3GNet** | Energy, Forces, Stresses | Multi-body interaction universal potential |

### Generative Discovery (GNoME-inspired)

Inspired by [Google DeepMind's GNoME](https://deepmind.google/discover/blog/millions-of-new-materials-discovered-with-deep-learning/) paper. Substitute elements in known stable materials and predict whether the hypothetical new compound is thermodynamically stable -- without synthesizing it in a lab.

```bash
matgraph substitute LiFePO4 Li Na
# Predicts: NaFePO4 stability vs LiFePO4
```

### XRD Simulation

Generate theoretical Cu-Ka X-Ray Diffraction patterns for any material. Useful for matching experimental peaks against predicted structures.

```bash
matgraph xrd LiFePO4
```

### Built-in Cache

All API responses and predictions are automatically cached in a local SQLite database (`~/.matgraph_cache/cache.db`). Repeated queries return instantly. No external service required.

```bash
matgraph cache stats    # View cache size and entry count
matgraph cache clear    # Wipe the cache
```

### API Key Authentication

Generate secure, multi-tenant API keys for the GraphQL server:

```bash
matgraph auth generate --user "research-team-A"
```

Keys are prefixed with `mg_`, stored in `~/.matgraph_keys.json`, and validated on every request. You can also set a master key via the `MATGRAPH_API_KEY` environment variable.

### Dataset Export

Export predictions to CSV or JSON for use in pandas, scikit-learn, or any ML pipeline. Optionally export 3D crystal structures as `.cif` files.

```bash
matgraph predict LiFePO4 --save results.json --format json --cif
```

---

## Architecture

```
matgraph/
  __init__.py      # Top-level SDK export
  sdk.py           # Python SDK (MatGraphSDK class)
  cli.py           # Typer CLI with Rich formatting
  core.py          # Pipeline orchestration and feature extraction
  cgcnn.py         # Crystal Graph Convolutional Neural Network
  megnet.py        # MatErials Graph Network
  m3gnet.py        # M3GNet Universal Potential
  graphql_app.py   # FastAPI + Strawberry GraphQL server
  auth.py          # API key generation and validation
  cdn.py           # SQLite cache layer
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
