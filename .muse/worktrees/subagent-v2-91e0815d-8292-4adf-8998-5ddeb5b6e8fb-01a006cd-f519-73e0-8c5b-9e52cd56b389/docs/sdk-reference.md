# Python SDK Reference

The `MatGraphSDK` class provides a clean Python interface for use in Jupyter Notebooks, scripts, and ML pipelines.

## Import

```python
from matgraph import MatGraphSDK
```

## Initialization

```python
sdk = MatGraphSDK(api_key="your_key")
```

If `api_key` is not provided, it reads from the `MP_API_KEY` environment variable.

---

## Methods

### sdk.predict()

Run the ML prediction pipeline on a material.

```python
sdk.predict(
    formula: str,
    model: str = "cgcnn",
    min_gap: float = None,
    max_gap: float = None,
    crystal_system: str = None
) -> List[Dict]
```

**Parameters:**
| Parameter | Type | Default | Description |
|---|---|---|---|
| `formula` | str | required | Chemical formula |
| `model` | str | `"cgcnn"` | `"cgcnn"`, `"megnet"`, or `"m3gnet"` |
| `min_gap` | float | None | Minimum band gap filter |
| `max_gap` | float | None | Maximum band gap filter |
| `crystal_system` | str | None | Crystal system filter |

**Returns:** List of dictionaries containing predictions and metadata.

**Example:**

```python
results = sdk.predict("LiFePO4", model="m3gnet")
for r in results:
    print(f"{r['formula']}: {r['m3gnet_energy']} eV")
```

---

### sdk.substitute()

Simulate elemental substitution and predict stability.

```python
sdk.substitute(
    formula: str,
    element_out: str,
    element_in: str
) -> Dict
```

**Parameters:**
| Parameter | Type | Description |
|---|---|---|
| `formula` | str | Base material formula |
| `element_out` | str | Element to remove |
| `element_in` | str | Element to insert |

**Returns:** Dictionary with `original`, `hypothetical`, and `is_more_stable` keys.

**Example:**

```python
result = sdk.substitute("LiFePO4", element_out="Li", element_in="Na")
if result["is_more_stable"]:
    print(f"{result['hypothetical']['formula']} is stable")
```

---

### sdk.xrd()

Simulate an X-Ray Diffraction pattern.

```python
sdk.xrd(formula: str) -> Dict
```

**Returns:** Dictionary with `two_theta`, `intensity`, and `hkls` arrays.

**Example:**

```python
xrd = sdk.xrd("LiFePO4")

# Plot with matplotlib
import matplotlib.pyplot as plt
plt.plot(xrd["two_theta"], xrd["intensity"])
plt.xlabel("2-theta (degrees)")
plt.ylabel("Intensity")
plt.title("XRD Pattern - LiFePO4")
plt.show()
```

---

### sdk.evaluate()

Compute Mean Absolute Error against ground truth.

```python
sdk.evaluate(formula: str, model: str = "cgcnn") -> Dict
```

**Returns:** Dictionary with `band_gap_mae`, `formation_energy_mae`, and `samples_evaluated`.

**Example:**

```python
metrics = sdk.evaluate("LiFePO4", model="megnet")
print(f"Band gap MAE: {metrics['band_gap_mae']:.4f} eV")
print(f"Formation energy MAE: {metrics['formation_energy_mae']:.4f} eV/atom")
print(f"Samples: {metrics['samples_evaluated']}")
```

---

## Common Workflows

### Batch Prediction

```python
formulas = ["LiFePO4", "NaCl", "SiO2", "TiO2", "GaN"]
for formula in formulas:
    results = sdk.predict(formula)
    for r in results:
        print(f"{r['formula']}: gap={r['predicted_band_gap']:.2f} eV")
```

### Screening Loop for New Materials

```python
import itertools

bases = ["LiFePO4", "LiCoO2", "LiMnO2"]
substitutions = [("Li", "Na"), ("Li", "K"), ("Li", "Mg")]

for base, (out, inp) in itertools.product(bases, substitutions):
    try:
        result = sdk.substitute(base, element_out=out, element_in=inp)
        status = "STABLE" if result["is_more_stable"] else "unstable"
        print(f"{base} -> {result['hypothetical']['formula']}: {status}")
    except ValueError:
        pass
```

### Export to Pandas DataFrame

```python
import pandas as pd

results = sdk.predict("LiFePO4")
df = pd.DataFrame([{
    "material_id": r["material_id"],
    "formula": r["formula"],
    "band_gap": r["predicted_band_gap"],
    "formation_energy": r["predicted_form_energy"],
    "density": r["features"]["density"],
} for r in results])

print(df)
df.to_csv("predictions.csv", index=False)
```
