# Installation

## Requirements

- Python 3.9 or higher
- A free [Materials Project](https://materialsproject.org/) API key — only
  for **live** predictions. The [demo mode](#try-it-in-60-seconds-no-key) needs nothing.

## Try it in 60 seconds (no key)

```bash
pip install matgraph-cli
matgraph predict Si --demo
matgraph xrd Si --demo
```

`--demo` runs on bundled sample data: real crystal structures and XRD
patterns, illustrative ML numbers, loudly labeled. No API key, no PyTorch,
no 2 GB download.

## Install options

### pip

```bash
pip install matgraph-cli            # slim base: CLI + cache + demo (~30 MB)
pip install "matgraph-cli[ml]"      # + full ML stack (torch, matgl, pymatgen...)
pip install "matgraph-cli[api]"     # + GraphQL/REST server
pip install "matgraph-cli[all]"     # everything
```

### uv (fastest)

```bash
uv tool install matgraph-cli
uv tool install "matgraph-cli[ml]"  # with ML stack
```

### conda

```bash
conda install -c conda-forge matgraph-cli
matgraph predict Si --demo            # works immediately
pip install "matgraph-cli[ml]"        # ML stack in the same env
```

(First release: the feedstock goes live after v3.0.0 — see
[`conda-recipe/README.md`](https://github.com/Himan-D/matgraph-cli/blob/main/conda-recipe/README.md).)

### Docker

```bash
docker build -t matgraph .
docker run -p 8000:8000 -e MP_API_KEY="$MP_API_KEY" matgraph
# GraphQL playground: http://localhost:8000/graphql
```

### From source

```bash
git clone https://github.com/Himan-D/matgraph-cli.git
cd matgraph-cli
uv sync
```

## Extras reference

| Extra | Adds | When you need it |
|---|---|---|
| *(base)* | CLI, cache, `--demo`, tracking, DFT writers | Trying out, scripting, CI |
| `mp` | pymatgen, mp-api | Live MP data without ML |
| `ml` | `mp` + torch, matgl, ase, scikit-learn | `predict`, `relax`, `train`, `evolve` |
| `api` | fastapi, strawberry, uvicorn | `matgraph serve` / GraphQL |
| `ui` | gradio | `matgraph track dashboard` |
| `tracking` | wandb | W&B cloud sync |
| `parquet` / `polars` | pandas+pyarrow / polars | DataFrame export |
| `quantum` | pennylane | `matgraph quantum ...` |
| `omat24` | `ml` + fairchem | EquiformerV2 potentials |
| `all` | everything | Full workstation install |

Missing something? Commands fail with the exact fix, e.g.
`pip install "matgraph-cli[ml]"` — never a bare `ModuleNotFoundError`.

## Authentication

Live predictions fetch structures from the Materials Project:

1. Create an account at [materialsproject.org](https://materialsproject.org/)
2. Copy your API key from the dashboard
3. `export MP_API_KEY="your_key_here"` (or `matgraph setup <KEY>`)

Add the export to `~/.bashrc` / `~/.zshrc` to make it permanent.

## Verify installation

```bash
matgraph --version
matgraph predict Si --demo     # proves the CLI works end to end
matgraph status                # keys, cache, models at a glance
```
