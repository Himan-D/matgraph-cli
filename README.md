# 🔬 MatGraph CLI & GraphQL API

**MatGraph** is the ultimate, open-source tool for Material Science researchers and Machine Learning engineers. It is a complete, production-ready product built for extreme usability and speed.

It abstracts away the complexity of the deep learning pipeline for material properties. With a single command or GraphQL query, you can:
1. **Fetch & Filter** high-fidelity crystal structures from the Materials Project with advanced constraints.
2. **Featurize** the materials extracting structural and compositional data.
3. **Predict** properties (like Band Gap) using built-in ML models.
4. **Save** datasets seamlessly to JSON or CSV.

## ✨ Features
- **Ultra-Fast Setup:** Powered by `uv` for lightning-fast dependency resolution.
- **Advanced CLI Filters:** Search by Band Gap (`--min-gap`, `--max-gap`) and Crystal System (`--crystal-system`).
- **Data Export:** Instantly save your ML predictions and feature sets using `--save data.csv --format csv`.
- **Modern GraphQL Engine:** Built with `Strawberry` & `FastAPI`. Fully asynchronous resolvers with nested metrics and filtering options.

---

## 🚀 Quick Start (Efficient with `uv`)

### 1. Installation

If you don't have `uv` installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repo and sync dependencies instantly:
```bash
git clone https://github.com/yourusername/matgraph-cli.git
cd matgraph-cli
uv sync
```

### 2. Signups and API Key (Important!)
You need an API key from the Materials Project:
1. Go to [Materials Project](https://materialsproject.org/)
2. Sign up / Log in and copy your API Key.
3. Set up your key in your environment:
```bash
export MP_API_KEY="YOUR_API_KEY"
```

---

## 🛠️ Usage: The Productive CLI

**Basic Prediction:**
```bash
uv run matgraph predict LiFePO4
```

**Advanced Filtering:**
Filter for materials with a minimum band gap of 1.5 eV and a cubic crystal system:
```bash
uv run matgraph predict LiFePO4 --min-gap 1.5 --crystal-system Cubic
```

**Export & Save Data:**
Save the extracted features and ML predictions directly to a dataset for offline training:
```bash
uv run matgraph predict LiFePO4 --min-gap 2.0 --save dataset.csv --format csv
```

---

## 🌐 Usage: The Modern GraphQL API

Spin up the async GraphQL server:
```bash
uv run matgraph serve --port 8000
```

**Example GraphQL Query with Filters:**
```graphql
query {
  predictMaterial(formula: "NaCl", minGap: 1.0, crystalSystem: "Cubic", limit: 3) {
    materialId
    formula
    crystalSystem
    trueBandGap
    predictedBandGap
    features {
      density
      numElements
    }
  }
}
```

---

## 🏗️ Tech Stack
- **Packaging:** uv (Astral) & Hatchling
- **CLI Framework:** Typer + Rich
- **GraphQL Engine:** Strawberry (Async) + FastAPI
- **Material Science:** PyMatGen + MP-API

## 🤝 Contributing
Ready for the open-source community! 
Run tests with `uv run pytest` and submit PRs for custom model integrations (like PyTorch CGCNN!).
