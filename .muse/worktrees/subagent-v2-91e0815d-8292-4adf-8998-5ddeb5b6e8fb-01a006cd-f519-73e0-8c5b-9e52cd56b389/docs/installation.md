# Installation

## Requirements

- Python 3.9 or higher
- A free Materials Project API key

## Install from PyPI

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is the fastest Python package manager. Install MatGraph as a global CLI tool:

```bash
uv tool install matgraph-cli
```

Or add it to an existing project:

```bash
uv add matgraph-cli
```

### Using pip

```bash
pip install matgraph-cli
```

## Install from Source

```bash
git clone https://github.com/Himan-D/matgraph-cli.git
cd matgraph-cli
uv sync
```

## Authentication

MatGraph fetches crystal structure data from the [Materials Project](https://materialsproject.org/) database. You need a free API key.

1. Create an account at [materialsproject.org](https://materialsproject.org/)
2. Go to your dashboard and copy your API key
3. Set it as an environment variable:

```bash
export MP_API_KEY="your_key_here"
```

To make it permanent, add this line to your `~/.bashrc`, `~/.zshrc`, or equivalent shell config.

## Verify Installation

```bash
matgraph --version
```

Expected output:

```
MatGraph CLI Version: 1.2.0
Python Version: 3.11.5
```
