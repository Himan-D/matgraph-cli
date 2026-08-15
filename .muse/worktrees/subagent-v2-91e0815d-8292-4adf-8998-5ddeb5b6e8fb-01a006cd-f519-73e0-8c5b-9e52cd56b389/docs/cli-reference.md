# CLI Reference

MatGraph CLI is built with [Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/) for a clean terminal experience.

## Global Options

```bash
matgraph --version    # Show MatGraph and Python version
matgraph --help       # Show all available commands
```

---

## Commands

### predict

Run the ML prediction pipeline on a material formula.

```bash
matgraph predict FORMULA [OPTIONS]
```

**Arguments:**
| Argument | Required | Description |
|---|---|---|
| `FORMULA` | Yes | Chemical formula (e.g., `LiFePO4`, `NaCl`, `SiO2`) |

**Options:**
| Option | Default | Description |
|---|---|---|
| `--model` | `cgcnn` | Model architecture: `cgcnn`, `megnet`, or `m3gnet` |
| `--min-gap` | None | Minimum band gap filter (eV) |
| `--max-gap` | None | Maximum band gap filter (eV) |
| `--crystal-system` | None | Filter by crystal system (e.g., `Cubic`, `Hexagonal`) |
| `--save` | None | Output file path for results |
| `--format` | `json` | Output format: `json` or `csv` |
| `--cif` | False | Export 3D crystal structures as `.cif` files |

**Examples:**

```bash
# Basic prediction with default CGCNN model
matgraph predict LiFePO4

# Use M3GNet for energy/forces/stresses
matgraph predict LiFePO4 --model m3gnet

# Filter by band gap range and crystal system
matgraph predict LiFePO4 --min-gap 1.5 --max-gap 4.0 --crystal-system Cubic

# Export results to CSV with CIF structures
matgraph predict LiFePO4 --save results.csv --format csv --cif
```

---

### substitute

Simulate elemental substitution and predict thermodynamic stability. Inspired by Google DeepMind's GNoME paper.

```bash
matgraph substitute FORMULA ELEMENT_OUT ELEMENT_IN
```

**Arguments:**
| Argument | Description |
|---|---|
| `FORMULA` | Base material formula |
| `ELEMENT_OUT` | Element to remove |
| `ELEMENT_IN` | Element to insert |

**Example:**

```bash
matgraph substitute LiFePO4 Li Na
# Predicts whether NaFePO4 is thermodynamically stable compared to LiFePO4
```

---

### xrd

Simulate the X-Ray Diffraction pattern for a material.

```bash
matgraph xrd FORMULA
```

**Example:**

```bash
matgraph xrd LiFePO4
# Shows top diffraction peaks with 2-theta angles and intensities
```

---

### evaluate

Evaluate model accuracy by computing Mean Absolute Error (MAE) against Materials Project ground truth.

```bash
matgraph evaluate FORMULA [--model MODEL]
```

**Example:**

```bash
matgraph evaluate LiFePO4 --model megnet
```

---

### serve

Start the GraphQL API server.

```bash
matgraph serve [--port PORT]
```

**Options:**
| Option | Default | Description |
|---|---|---|
| `--port` | `8000` | Port to run the server on |

---

### auth generate

Generate an API key for the GraphQL server.

```bash
matgraph auth generate --user USERNAME
```

**Example:**

```bash
matgraph auth generate --user "research-team"
# Output: mg_S8jvhzo58p6XQE_fS0Oru1WqG2AJR6x5
```

---

### cache stats

Show cache statistics.

```bash
matgraph cache stats
```

Output:

```
Entries: 12
Size: 0.03 MB
Location: /Users/you/.matgraph_cache/cache.db
```

### cache clear

Clear all cached results.

```bash
matgraph cache clear
```
