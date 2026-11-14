"""MatGraph live demo for Hugging Face Spaces — zero-config, no API key.

Runs entirely on bundled demo data (MatGraphSDK(demo=True)): real crystal
structures/XRD plus illustrative ML samples, honestly labeled. Free CPU
hardware is enough.
"""

import gradio as gr
import pandas as pd

from matgraph import MatGraphSDK
from matgraph.demo import DEMO_FORMULAS

sdk = MatGraphSDK(demo=True)


def predict_material(formula: str, model: str):
    """Sample prediction table for one formula."""
    try:
        rows = sdk.predict(formula, model=model)
    except Exception as e:  # unknown formula etc. — show the message, not a traceback
        return pd.DataFrame([{"Error": str(e)}]), "*Try LiFePO4, Si, or NaCl.*"
    df = pd.DataFrame(
        [
            {
                "ID": r["material_id"],
                "Formula": r["formula"],
                "Crystal System": r["crystal_system"],
                "True Gap (eV)": r["true_band_gap"],
                "Pred Gap (eV)": r["predicted_band_gap"],
                "True E_form": r["true_form_energy"],
                "Pred E_form": r["predicted_form_energy"],
                "Energy (eV)": round(r["m3gnet_energy"], 3),
            }
            for r in rows
        ]
    )
    caption = (
        "**Demo sample** — structure/XRD are real crystallography; ML numbers are "
        "illustrative. For live predictions: `pip install \"matgraph-cli[ml]\"` + a free "
        "[Materials Project](https://materialsproject.org/) API key."
    )
    return df, caption


def xrd_pattern(formula: str):
    """Real Cu-Ka XRD pattern as an interactive plot + top-peaks table."""
    try:
        data = sdk.xrd(formula)
    except Exception as e:
        return pd.DataFrame([{"Error": str(e)}]), pd.DataFrame([{"Error": str(e)}])
    plot_df = pd.DataFrame({"2θ (deg)": data["two_theta"], "Intensity (%)": data["intensity"]})
    peaks = sorted(
        zip(data["two_theta"], data["intensity"], data["hkls"]), key=lambda p: -p[1]
    )[:10]
    table_df = pd.DataFrame(
        [{"2θ (deg)": round(t, 2), "Intensity (%)": i, "hkl": str(h[0])} for t, i, h in peaks]
    )
    return plot_df, table_df


def battery_screen(formula: str):
    """Sample cathode screening (exact capacity, sample voltage)."""
    try:
        m = sdk.battery(formula)
    except Exception as e:
        return f"Error: {e}"
    return (
        f"**{m['formula']}** (carrier: {m['carrier']})\n\n"
        f"- Theoretical capacity: **{m['theoretical_capacity_mah_g']} mAh/g** (exact)\n"
        f"- Avg voltage: **{m['avg_voltage_V_proxy']} V** (ML estimate, sample)\n\n"
        f"*{m['reference']}*"
    )


def substitute_material(formula: str, elem_out: str, elem_in: str):
    """Sample substitution analysis."""
    try:
        res = sdk.substitute(formula, elem_out, elem_in)
    except Exception as e:
        return f"Error: {e}"
    verdict = (
        "MORE stable (sample)"
        if res["is_more_stable"]
        else "LESS stable (sample — validate with DFT)"
    )
    return (
        f"**{res['original']['formula']}**: {res['original']['energy']:.3f} eV → "
        f"**{res['hypothetical']['formula']}**: {res['hypothetical']['energy']:.3f} eV\n\n"
        f"Verdict: hypothetical is {verdict}."
    )


with gr.Blocks(title="MatGraph — Materials ML Demo") as demo:
    gr.Markdown("# MatGraph — Deep Learning for Materials Science")
    gr.Markdown(
        "Predict properties, simulate XRD, and triage hypothetical compounds. "
        "This Space runs on **bundled demo data** (no API key) — "
        "[install MatGraph](https://pypi.org/project/matgraph-cli/) for live predictions. "
        "If this saved you time, star "
        "[Himan-D/matgraph-cli](https://github.com/Himan-D/matgraph-cli)!"
    )

    with gr.Tab("Predict"):
        with gr.Row():
            formula = gr.Dropdown(list(DEMO_FORMULAS), value="LiFePO4", label="Formula")
            model = gr.Dropdown(
                ["m3gnet", "megnet", "cgcnn", "chgnet"], value="m3gnet", label="Model"
            )
        btn = gr.Button("Predict", variant="primary")
        out_table = gr.Dataframe(label="Sample prediction")
        out_caption = gr.Markdown()
        btn.click(predict_material, [formula, model], [out_table, out_caption])

    with gr.Tab("XRD"):
        gr.Markdown("Real Cu-Ka diffraction patterns computed from crystal structures.")
        formula_x = gr.Dropdown(list(DEMO_FORMULAS), value="Si", label="Formula")
        btn_x = gr.Button("Simulate XRD", variant="primary")
        plot = gr.LinePlot(label="XRD pattern")
        peaks_table = gr.Dataframe(label="Top 10 peaks")
        btn_x.click(xrd_pattern, [formula_x], [plot, peaks_table])

    with gr.Tab("Battery"):
        gr.Markdown("Cathode screening: exact theoretical capacity + ML voltage estimate.")
        formula_b = gr.Dropdown(list(DEMO_FORMULAS), value="LiFePO4", label="Formula")
        btn_b = gr.Button("Screen cathode", variant="primary")
        res_b = gr.Markdown()
        btn_b.click(battery_screen, [formula_b], [res_b])

    with gr.Tab("Substitute"):
        gr.Markdown("Triage a hypothetical elemental substitution (sample energies).")
        with gr.Row():
            formula_s = gr.Dropdown(list(DEMO_FORMULAS), value="LiFePO4", label="Formula")
            out_s = gr.Dropdown(
                ["Li", "Na", "Fe", "Si", "Cl", "O", "P"], value="Li", label="Replace"
            )
            in_s = gr.Dropdown(
                ["Li", "Na", "K", "Fe", "Ge", "Cl", "O"], value="Na", label="With"
            )
        btn_s = gr.Button("Analyze stability", variant="primary")
        res_s = gr.Markdown()
        btn_s.click(substitute_material, [formula_s, out_s, in_s], [res_s])

if __name__ == "__main__":
    demo.launch()
