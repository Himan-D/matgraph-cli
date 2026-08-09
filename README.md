<div align="center">
  <h1>MatGraph</h1>
  <p><strong>The modern, end-to-end Material Science Deep Learning Pipeline & GraphQL API</strong></p>
  
  [![PyPI - Version](https://img.shields.io/pypi/v/matgraph-cli?color=blue)](https://pypi.org/project/matgraph-cli/)
  [![Python Versions](https://img.shields.io/pypi/pyversions/matgraph-cli)](https://pypi.org/project/matgraph-cli/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

---

## Key Features (v0.6.0 Update)
*   **Generative Discovery (GNoME-inspired):** Simulate elemental substitution (e.g., swapping Li for Na) and predict thermodynamic stability to discover new materials.
*   **Universal Interatomic Potentials (M3GNet):** New `m3gnet` architecture for predicting Energy, Forces, and Stresses using simulated 3-body interactions.
*   **X-Ray Diffraction (XRD) Simulator:** Generate synthetic Cu-Kα XRD patterns directly from the CLI to identify material peaks.
*   **Analytics & Evaluation:** `evaluate` command computes Mean Absolute Error (MAE) comparing PyTorch predictions directly against Materials Project ground truth.
*   **Raw Structure Export:** Automatically export raw 3D crystal structures to standard `.cif` formats.
*   **Multi-Architecture Support:** Seamlessly switch between CGCNN, MEGNet, and M3GNet models.

---

## Installation

We recommend using [**uv**](https://github.com/astral-sh/uv) for the fastest installation experience.

```bash
# Install via uv (Recommended)
uv tool install matgraph-cli

# Or via standard pip
pip install matgraph-cli
```

### Authentication Setup
To fetch high-fidelity data, you need a free API key from the [Materials Project](https://materialsproject.org/).
```bash
export MP_API_KEY="your_api_key_here"
```

---

## Usage: The Productive CLI

MatGraph's CLI is designed to be highly intuitive. 

**1. Generative Discovery (Elemental Substitution)**
Inspired by DeepMind's GNoME, substitute elements in a known material and predict if the new hypothetical crystal will be thermodynamically stable:
```bash
matgraph substitute LiFePO4 Li Na
```

**2. X-Ray Diffraction (XRD) Simulation**
Generate the theoretical XRD pattern (top peaks and intensities) for any material:
```bash
matgraph xrd LiFePO4
```

**3. Universal Interatomic Potentials (M3GNet)**
Use the M3GNet architecture to predict structural energy, forces, and stresses:
```bash
matgraph predict LiFePO4 --model m3gnet
```

**4. Analytics & Model Evaluation**
Evaluate the accuracy (MAE) of a specific architecture against true scientific data:
```bash
matgraph evaluate LiFePO4 --model megnet
```

**5. Advanced Search & Filtering**
Filter materials based on physical constraints:
```bash
matgraph predict LiFePO4 --min-gap 1.5 --crystal-system Cubic --model megnet
```

**6. Structure & Dataset Export for ML Engineers**
Save extracted predictions directly to a dataset (CSV/JSON), and export 3D `.cif` files for offline processing:
```bash
matgraph predict LiFePO4 --min-gap 2.0 --save dataset.csv --format csv --cif
```

---

## Usage: Python SDK (For Jupyter Notebooks)

MatGraph provides a powerful Python SDK for seamless integration into Jupyter Notebooks, Pandas workflows, or custom backend services.

```python
from matgraph import MatGraphSDK

# Initialize SDK (Auto-loads MP_API_KEY from env if not provided)
sdk = MatGraphSDK()

# 1. Predict properties using Deep Learning
results = sdk.predict("LiFePO4", model="m3gnet")
print(f"Predicted Energy: {results[0]['m3gnet_energy']} eV")

# 2. Generative Discovery (Substitution Analysis)
discovery = sdk.substitute("LiFePO4", element_out="Li", element_in="Na")
if discovery["is_more_stable"]:
    print("New material is stable!")

# 3. Simulate XRD Patterns
xrd_data = sdk.xrd("LiFePO4")
print(f"Top Peak Angle: {xrd_data['two_theta'][0]}")

# 4. Model Analytics
metrics = sdk.evaluate("LiFePO4", model="megnet")
print(f"MAE: {metrics['band_gap_mae']}")
```

---

## Usage: The Modern GraphQL API

Integrate MatGraph into your own web applications seamlessly using our robust, async GraphQL engine. **Note:** The API is now secured with an API Key mechanism.

### Start the Server:
```bash
uvicorn matgraph.graphql_app:app --reload
```

### Security (API Key Generation)
The API is secured. You can generate a multi-tenant secure API key for different users from the CLI:

```bash
matgraph auth generate --user "Client-A"
```
*Output: `mg_S8jvhzo58p6XQE_fS0Oru1WqG2AJR6x5`*

This key will be saved locally (`~/.matgraph_keys.json`) and instantly authorized to use your GraphQL API.

Alternatively, you can set a master API Key using an environment variable on your server:
```bash
export MATGRAPH_API_KEY="my_super_secret_key"
```

### Sample Query (cURL):
```bash
curl -X POST http://localhost:8000/graphql \
  -H "Content-Type: application/json" \
  -H "x-api-key: <YOUR_GENERATED_KEY>" \
  -d '{"query": "query { predictMaterial(formula: \"LiFePO4\") { predictedFormEnergy } }"}'
```

Navigate to `http://localhost:8000/graphql` to explore the interactive GraphiQL playground.

```graphql
query {
  predictMaterial(formula: "NaCl", minGap: 1.0, limit: 3, model: "megnet") {
    materialId
    formula
    crystalSystem
    trueBandGap
    predictedBandGap
    trueFormEnergy
    predictedFormEnergy
    features {
      density
      numElements
      volume
    }
    metrics {
      modelName
      confidenceScore
    }
  }
}
```

---

## Releases & Changelog

### **v0.3.x (Current - Multi-Property & MEGNet Update)**
*   **Feature:** Implemented PyTorch MEGNet model (`matgraph/megnet.py`).
*   **Feature:** Support for Formation Energy predictions alongside Band Gap.
*   **Feature:** Added `--model megnet` flag and GraphQL `model: "megnet"` argument.
*   **Feature:** Integrated PyTorch architecture (`CrystalGraphConvNet`) replacing legacy dummy models.
*   **Feature:** Advanced CLI filtering (`--min-gap`, `--max-gap`, `--crystal-system`).
*   **Feature:** One-command dataset exporting (`--save`, `--format`).
*   **Improvement:** GraphQL schema modernized with detailed `ModelMetrics` and GraphQL pagination filters.

### **v0.1.x (Initial Release)**
*   Initial end-to-end pipeline with MP-API fetching and basic feature extraction.
*   GraphQL Server & basic Typer CLI introduced.
*   Project migrated to `uv` build backend for maximum efficiency.

---

## Contributing & Architecture
MatGraph is built on a robust, modern Python stack:
*   **ML & Science:** PyTorch, PyMatGen, Scikit-Learn, MP-API
*   **API & CLI:** FastAPI, Strawberry GraphQL, Typer, Rich
*   **Packaging:** uv (Hatchling)

We welcome contributions! To set up for local development:
```bash
git clone https://github.com/Himan-D/matgraph-cli.git
cd matgraph-cli
uv sync
uv run pytest
```
Please open an issue before submitting major pull requests.
