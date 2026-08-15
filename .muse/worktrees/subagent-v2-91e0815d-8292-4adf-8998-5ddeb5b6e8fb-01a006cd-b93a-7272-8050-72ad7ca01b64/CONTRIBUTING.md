# Contributing to MatGraph

Thank you for your interest in contributing to MatGraph. This guide will help you get started.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- A free [Materials Project](https://materialsproject.org/) API key

### Local Development Setup

```bash
git clone https://github.com/Himan-D/matgraph-cli.git
cd matgraph-cli
uv sync
export MP_API_KEY="your_key_here"
```

### Running Tests

```bash
uv run pytest
```

### Running the CLI Locally

```bash
uv run matgraph predict LiFePO4
uv run matgraph --version
```

---

## How to Contribute

### Reporting Bugs

Open a [GitHub Issue](https://github.com/Himan-D/matgraph-cli/issues/new) with:

- A clear title describing the problem
- Steps to reproduce the issue
- Expected vs actual behavior
- Your Python version and OS
- The full error traceback if applicable

### Suggesting Features

Open an issue with the title prefixed by `[Feature]`. Include:

- What problem does it solve?
- Who benefits from it (researchers, ML engineers, developers)?
- Example usage (CLI command, SDK call, or API query)

### Submitting Code

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run tests: `uv run pytest`
5. Commit with a clear message: `git commit -m "feat: add phonon dispersion calculation"`
6. Push to your fork: `git push origin feature/your-feature-name`
7. Open a Pull Request against `main`

---

## What to Work On

### Good First Issues

If you are new to the project, look for issues labeled `good first issue`. Here are some areas that are always welcome:

- Adding docstrings to functions that lack them
- Writing unit tests for existing modules
- Fixing typos in documentation
- Adding examples to the docs

### High-Impact Areas

| Area | Description | Files |
|---|---|---|
| New GNN architectures | Add SchNet, DimeNet, ALIGNN, etc. | `matgraph/` |
| Phase diagram support | Compute binary/ternary phase diagrams | `matgraph/core.py` |
| Phonon calculations | Predict phonon density of states | `matgraph/core.py` |
| Bandstructure plotting | Visualize electronic band structures | new module |
| REST API endpoints | Add REST alongside GraphQL | `matgraph/graphql_app.py` |
| Pre-trained model weights | Host and load trained checkpoints | `matgraph/cgcnn.py`, etc. |
| Batch processing | Process multiple formulas in one call | `matgraph/sdk.py`, `matgraph/cli.py` |
| Database adapters | Support AFLOW, OQMD, JARVIS beyond MP | `matgraph/core.py` |

### Documentation

Documentation improvements are always appreciated:

- Tutorials for common workflows
- API reference pages
- Jupyter Notebook examples
- Blog posts or videos about MatGraph

---

## Code Style

- Use type hints for all function signatures
- Write docstrings for all public functions
- Keep functions focused and under 50 lines where possible
- Follow existing patterns in the codebase
- No emojis or icons in code or docs

---

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add phonon dispersion calculation
fix: correct XRD peak intensity normalization
docs: add SDK usage tutorial
test: add unit tests for substitute_material
refactor: extract feature engineering into separate module
```

---

## Project Structure

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

---

## Community

- Open an issue for questions or discussion
- Tag PRs with relevant labels
- Be respectful and constructive in all interactions

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
