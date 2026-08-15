# Tutorials

Practical walkthroughs for common MatGraph workflows.

---

## Tutorial 1: Screen Battery Cathode Materials

Find which alkali metal substitutions in LiFePO4 produce stable cathode candidates.

```python
from matgraph import MatGraphSDK

sdk = MatGraphSDK()

base = "LiFePO4"
candidates = ["Na", "K", "Mg", "Ca", "Zn"]

print(f"Screening substitutions for {base}")
print("-" * 50)

for element in candidates:
    try:
        result = sdk.substitute(base, element_out="Li", element_in=element)
        new_formula = result["hypothetical"]["formula"]
        stable = "STABLE" if result["is_more_stable"] else "UNSTABLE"
        delta_e = result["hypothetical"]["energy"] - result["original"]["energy"]
        print(f"  {base} -> {new_formula}: {stable} (delta E = {delta_e:.4f} eV)")
    except ValueError as e:
        print(f"  {element}: skipped ({e})")
```

---

## Tutorial 2: Compare Model Architectures

Evaluate all three models on the same material and compare accuracy.

```python
from matgraph import MatGraphSDK

sdk = MatGraphSDK()

formula = "SiO2"
models = ["cgcnn", "megnet"]

print(f"Model comparison for {formula}")
print("-" * 40)

for model in models:
    metrics = sdk.evaluate(formula, model=model)
    print(f"  {model.upper()}")
    print(f"    Band gap MAE:    {metrics['band_gap_mae']:.4f} eV")
    print(f"    Form energy MAE: {metrics['formation_energy_mae']:.4f} eV/atom")
    print(f"    Samples:         {metrics['samples_evaluated']}")
    print()
```

---

## Tutorial 3: Build an XRD Database

Generate XRD patterns for a list of materials and save to a JSON file.

```python
import json
from matgraph import MatGraphSDK

sdk = MatGraphSDK()

materials = ["LiFePO4", "NaCl", "TiO2", "SiO2", "GaN", "ZnO"]
xrd_database = {}

for formula in materials:
    try:
        xrd = sdk.xrd(formula)
        xrd_database[formula] = {
            "num_peaks": len(xrd["two_theta"]),
            "strongest_peak_angle": xrd["two_theta"][xrd["intensity"].index(max(xrd["intensity"]))],
            "pattern": xrd
        }
        print(f"  {formula}: {len(xrd['two_theta'])} peaks")
    except Exception as e:
        print(f"  {formula}: failed ({e})")

with open("xrd_database.json", "w") as f:
    json.dump(xrd_database, f, indent=2)

print(f"\nSaved {len(xrd_database)} patterns to xrd_database.json")
```

---

## Tutorial 4: Export Predictions to Pandas

```python
import pandas as pd
from matgraph import MatGraphSDK

sdk = MatGraphSDK()

formulas = ["LiFePO4", "NaCl", "SiO2", "TiO2", "GaN"]
rows = []

for formula in formulas:
    results = sdk.predict(formula)
    for r in results:
        rows.append({
            "material_id": r["material_id"],
            "formula": r["formula"],
            "crystal_system": r["crystal_system"],
            "band_gap_true": r["true_band_gap"],
            "band_gap_pred": r["predicted_band_gap"],
            "form_energy_true": r["true_form_energy"],
            "form_energy_pred": r["predicted_form_energy"],
            "density": r["features"]["density"],
            "volume": r["features"]["volume"],
        })

df = pd.DataFrame(rows)
print(df.describe())
df.to_csv("matgraph_dataset.csv", index=False)
```

---

## Tutorial 5: Build a Web Dashboard

Use the GraphQL API to power a React or Next.js dashboard.

### Backend (already running)

```bash
matgraph auth generate --user "dashboard"
uvicorn matgraph.graphql_app:app --reload
```

### Frontend (JavaScript)

```javascript
async function predictMaterial(formula) {
  const response = await fetch("http://localhost:8000/graphql", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": "mg_your_key_here",
    },
    body: JSON.stringify({
      query: `
        query($formula: String!) {
          predictMaterial(formula: $formula) {
            formula
            crystalSystem
            predictedBandGap
            predictedFormEnergy
            features { density volume }
          }
        }
      `,
      variables: { formula },
    }),
  });
  return response.json();
}

// Usage
predictMaterial("LiFePO4").then(data => console.log(data));
```
