import gradio as gr
from matgraph.sdk import MatGraphSDK
import pandas as pd

def predict_material(formula, api_key):
    if not api_key:
        return "Please enter your Materials Project API Key."
    
    try:
        sdk = MatGraphSDK(api_key=api_key)
        results = sdk.predict(formula=formula, model="m3gnet")
        
        if not results:
            return "No data found for this formula."
            
        # Format results into a dataframe
        df = pd.DataFrame([{
            "ID": r["material_id"],
            "Formula": r["formula"],
            "Crystal System": r["crystal_system"],
            "Predicted Energy (eV)": round(r["m3gnet_energy"], 3) if r.get("m3gnet_energy") else "N/A"
        } for r in results])
        
        return df
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks(title="MatGraph CLI: Deep Learning for Material Science", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔬 MatGraph Explorer")
    gr.Markdown("Predict thermodynamic stability and properties of materials using M3GNet Universal Potentials.")
    
    with gr.Row():
        with gr.Column():
            formula_input = gr.Textbox(label="Chemical Formula (e.g., LiFePO4)", placeholder="LiFePO4")
            api_input = gr.Textbox(label="Materials Project API Key", type="password")
            btn = gr.Button("Predict Properties", variant="primary")
            
        with gr.Column():
            output_table = gr.Dataframe(label="Polymorph Predictions")
            
    btn.click(fn=predict_material, inputs=[formula_input, api_input], outputs=[output_table])
    
    gr.Markdown("Powered by [matgraph-cli](https://pypi.org/project/matgraph-cli/)")

demo.launch()
