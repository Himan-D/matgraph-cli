# MatGraph vs raw pymatgen + MatGL

MatGraph is a **workflow layer** on top of pymatgen, MatGL, and `mp-api` —
not a replacement. If you already script those libraries, here is exactly
what MatGraph saves you, task by task.

## 1. Formation energy for one formula

**Raw stack** — fetch, load weights, infer, all by hand:

```python
from mp_api.client import MPRester
import matgl

with MPRester(API_KEY) as mpr:
    docs = mpr.materials.summary.search(
        formula="LiFePO4",
        fields=["material_id", "structure", "formation_energy_per_atom"],
    )
model = matgl.load_model("M3GNet-Eform-MP-2019.4.1")  # downloads weights
for doc in docs:
    if doc.structure is None:
        continue
    eform = float(model.predict_structure(doc.structure).detach().item())
    print(doc.material_id, eform)
```

**MatGraph** — one command, with cache + provenance + seed handling built in:

```bash
matgraph predict LiFePO4 --model m3gnet --seed 42
```

```python
from matgraph import MatGraphSDK
sdk = MatGraphSDK()
sdk.predict("LiFePO4", model="m3gnet", seed=42)
```

## 2. Screen hypothetical substitutions

**Raw stack** — loop over polymorphs, mutate structures, run the PES:

```python
from mp_api.client import MPRester
from matgl.ext.ase import M3GNetCalculator
from pymatgen.io.ase import AseAtomsAdaptor
import matgl

pes = matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2")
with MPRester(API_KEY) as mpr:
    docs = mpr.materials.summary.search(formula="LiFePO4", fields=["structure"])
base = docs[0].structure
new = base.copy()
new.replace_species({"Li": "Na"})  # hope the site order cooperates
for label, s in [("base", base), ("hypo", new)]:
    atoms = AseAtomsAdaptor.get_atoms(s)
    atoms.calc = M3GNetCalculator(potential=pes)
    print(label, atoms.get_potential_energy())
```

**MatGraph** — validated, deterministic, tabulated:

```bash
matgraph substitute LiFePO4 Li Na --seed 42
```

```python
sdk.substitute("LiFePO4", "Li", "Na", seed=42)
```

## 3. Battery cathode triage

**Raw stack** — parse the composition, look up the Faraday constant,
decide on a voltage model, wire it to ML output yourself.

**MatGraph**:

```bash
matgraph battery LiFePO4            # exact capacity, keyless with [mp]
matgraph battery LiFePO4 --demo     # zero-config sample
```

```python
sdk.battery("LiFePO4")
# {'theoretical_capacity_mah_g': 169.89, 'avg_voltage_V_proxy': ..., ...}
```

## 4. Serve predictions to an app

**Raw stack** — write a FastAPI service, auth, validation, and error
handling from scratch.

**MatGraph**:

```bash
pip install "matgraph-cli[api]"
matgraph auth generate --user "my-app"
matgraph serve   # GraphQL + POST /v1/predict, hashed keys, validation
```

## What you get beyond fewer lines

| Concern | Raw stack | MatGraph |
|---|---|---|
| Repeated queries | Re-fetch + re-infer every time | Content-hashed SQLite cache |
| Reproducibility | Your own bookkeeping | `provenance` on every result (checkpoint, versions, seed, device) |
| Determinism | Manual seeding per lib | One `--seed` threads through all libs |
| Batch + export | Custom loops | `predict_many`, CSV/JSON/Parquet, `--cif` |
| Experiment tracking | Print statements | W&B-compatible local tracking + registry |
| Newcomer onboarding | Key + torch + weights first | `--demo` works in 60 seconds, no key |

## When to stay raw

- You need a model MatGraph doesn't wrap — use the libraries directly.
- You are developing a new potential — MatGraph consumes checkpoints, it
  doesn't train foundation models (its `train` fits small task heads).
- You need per-atom control of the relaxation loop — use ASE + MatGL.

Rule of thumb: libraries for research into models, MatGraph for research
with models.
