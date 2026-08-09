import typer
from rich.console import Console
from rich.table import Table
import os
import uvicorn
from typing import Optional
from matgraph.core import run_pipeline, save_results, substitute_material, simulate_xrd
from matgraph.sdk import MatGraphSDK
from matgraph.auth import generate_api_key

import sys
import importlib.metadata

app = typer.Typer(help="MatGraph CLI: Deep Learning for Material Science", no_args_is_help=True)
api_app = typer.Typer(help="Manage API keys for the GraphQL Server")
app.add_typer(api_app, name="auth")
console = Console()

@api_app.command("generate")
def generate_key(user: str = typer.Option(..., "--user", "-u", help="Username or identifier for this API key")):
    """Generates a secure API key for authenticating with the MatGraph GraphQL Server."""
    console.print(f"[bold cyan]Generating API Key for {user}...[/bold cyan]")
    key = generate_api_key(user)
    console.print(f"[bold green]Success![/bold green] Your API Key is: [bold yellow]{key}[/bold yellow]")
    console.print("[bold red]Please save this key securely! It grants access to the GraphQL API.[/bold red]")

def version_callback(value: bool):
    if value:
        try:
            version = importlib.metadata.version("matgraph-cli")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        console.print(f"MatGraph CLI Version: [bold green]{version}[/bold green]")
        console.print(f"Python Version: [bold cyan]{sys.version.split()[0]}[/bold cyan]")
        raise typer.Exit()

@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True, help="Show the application and Python version."
    )
):
    pass

@app.command()
def setup(api_key: str):
    """Set up your Materials Project API Key."""
    console.print(f"[green]Setup Instructions:[/green]")
    console.print("To use the CLI and API, export your API key in your terminal:")
    console.print(f'[bold cyan]export MP_API_KEY="{api_key}"[/bold cyan]')

@app.command()
def predict(
    formula: str, 
    min_gap: Optional[float] = typer.Option(None, "--min-gap", help="Minimum true band gap to filter (eV)"),
    max_gap: Optional[float] = typer.Option(None, "--max-gap", help="Maximum true band gap to filter (eV)"),
    crystal_system: Optional[str] = typer.Option(None, "--crystal-system", help="Filter by crystal system (e.g., Cubic, Hexagonal)"),
    save: Optional[str] = typer.Option(None, "--save", help="File path to save results (e.g., results.csv)"),
    format: str = typer.Option("csv", "--format", help="Save format: 'csv' or 'json'"),
    model: str = typer.Option("cgcnn", "--model", help="Model to use for prediction: 'cgcnn' or 'megnet'"),
    cif: bool = typer.Option(False, "--cif", help="Export the raw crystal structure of the results to .cif files")
):
    """Run the complete ML pipeline with advanced search filters and data saving."""
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        console.print("[red]Error: MP_API_KEY is not set.[/red]")
        raise typer.Exit(code=1)
    
    console.print(f"[cyan]Searching and running ML pipeline for {formula} using {model.upper()}...[/cyan]")
    if min_gap or max_gap or crystal_system:
        console.print(f"[dim]Filters applied - Min Gap: {min_gap}, Max Gap: {max_gap}, System: {crystal_system}[/dim]")
    
    try:
        results = run_pipeline(
            formula=formula, 
            api_key=api_key, 
            min_gap=min_gap, 
            max_gap=max_gap, 
            crystal_system=crystal_system,
            model=model
        )
        
        if not results:
            console.print(f"[yellow]No materials found matching the criteria.[/yellow]")
            return
            
        table = Table(title=f"Prediction Results for {formula} ({model.upper()})")
        table.add_column("ID", style="cyan")
        table.add_column("Formula", style="magenta")
        
        if model.lower() == "m3gnet":
            table.add_column("Energy (eV)", style="bold blue")
            table.add_column("Max Force", style="green")
            table.add_column("Crystal Sys", style="yellow")
        else:
            table.add_column("True Gap (eV)", style="green")
            table.add_column("Pred Gap", style="bold blue")
            table.add_column("True Form (eV)", style="green")
            table.add_column("Pred Form", style="bold blue")
        
        for r in results:
            if model.lower() == "m3gnet":
                energy = f"{r['m3gnet_energy']:.3f}" if r['m3gnet_energy'] is not None else "N/A"
                max_force = f"{max(r['m3gnet_forces']):.3f}" if r['m3gnet_forces'] else "N/A"
                table.add_row(str(r["material_id"]).replace("mp-", ""), r["formula"], energy, max_force, r["crystal_system"])
            else:
                true_form = str(round(r["true_form_energy"], 3)) if r["true_form_energy"] is not None else "N/A"
                true_gap = str(round(r["true_band_gap"], 3)) if r["true_band_gap"] is not None else "N/A"
                pred_gap = str(round(r["predicted_band_gap"], 3)) if r["predicted_band_gap"] is not None else "N/A"
                pred_form = str(round(r["predicted_form_energy"], 3)) if r["predicted_form_energy"] is not None else "N/A"
                
                table.add_row(
                    str(r["material_id"]).replace("mp-", ""),
                    r["formula"],
                    true_gap,
                    pred_gap,
                    true_form,
                    pred_form
                )
            
            if cif and r.get("structure"):
                cif_filename = f"{r['material_id']}_{r['formula']}.cif"
                r["structure"].to(filename=cif_filename)
                console.print(f"[dim]Exported structure to {cif_filename}[/dim]")
            
        console.print(table)
        console.print(f"[green]Pipeline completed successfully! Found {len(results)} items.[/green]")
        
        if save:
            save_results(results, save, format.lower())
            console.print(f"[bold green]Saved {len(results)} results to {save} (Format: {format.upper()})[/bold green]")

    except Exception as e:
        console.print(f"[red]Pipeline Error: {e}[/red]")

@app.command()
def xrd(formula: str):
    """Simulate X-Ray Diffraction (XRD) patterns for materials."""
    from matgraph.core import fetch_materials_data, simulate_xrd
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        console.print("[red]Error: MP_API_KEY is not set.[/red]")
        raise typer.Exit(code=1)
        
    console.print(f"[cyan]Fetching structures and simulating XRD for {formula}...[/cyan]")
    try:
        docs = fetch_materials_data(formula, api_key)
        if not docs:
            console.print("[yellow]No materials found.[/yellow]")
            return
            
        # Simulate XRD for the best match
        doc = docs[0]
        if not doc.structure:
            console.print("[red]No structure available to simulate XRD.[/red]")
            return
            
        xrd_data = simulate_xrd(doc.structure)
        
        table = Table(title=f"XRD Peaks (Cu-Ka) for {doc.formula_pretty} ({doc.material_id})")
        table.add_column("2θ (degrees)", justify="right", style="cyan")
        table.add_column("Intensity (%)", justify="right", style="green")
        table.add_column("hkl", style="magenta")
        
        # Display top 10 peaks
        peaks = list(zip(xrd_data["two_theta"], xrd_data["intensity"], xrd_data["hkls"]))
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        for theta, intensity, hkl in peaks[:10]:
            hkl_str = str(hkl[0]) # take the primary hkl family
            table.add_row(f"{theta:.2f}", f"{intensity:.1f}", hkl_str)
            
        console.print(table)
        console.print(f"[dim]Showing top 10 peaks. Use python API for full pattern export.[/dim]")
    except Exception as e:
        console.print(f"[red]XRD Error: {e}[/red]")

@app.command()
def evaluate(formula: str, model: str = typer.Option("cgcnn", "--model", help="Model to evaluate")):
    """Evaluate model accuracy (MAE) against true Materials Project data."""
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        console.print("[red]Error: MP_API_KEY is not set.[/red]")
        raise typer.Exit(code=1)
        
    console.print(f"[cyan]Evaluating {model.upper()} accuracy for {formula} variants...[/cyan]")
    try:
        results = run_pipeline(formula=formula, api_key=api_key, model=model)
        
        if not results:
            console.print("[yellow]No data found for evaluation.[/yellow]")
            return
            
        gap_errors = []
        form_errors = []
        
        for r in results:
            if r.get("true_band_gap") is not None and r.get("predicted_band_gap") is not None:
                gap_errors.append(abs(r["true_band_gap"] - r["predicted_band_gap"]))
            if r.get("true_form_energy") is not None and r.get("predicted_form_energy") is not None:
                form_errors.append(abs(r["true_form_energy"] - r["predicted_form_energy"]))
                
        gap_mae = sum(gap_errors) / len(gap_errors) if gap_errors else 0.0
        form_mae = sum(form_errors) / len(form_errors) if form_errors else 0.0
        
        console.print(f"\n[bold magenta]Model Analytics Report ({model.upper()})[/bold magenta]")
        console.print(f"Total Samples Evaluated: {len(results)}")
        if gap_errors:
            console.print(f"Band Gap MAE: [bold yellow]{gap_mae:.3f} eV[/bold yellow]")
        if form_errors:
            console.print(f"Formation Energy MAE: [bold yellow]{form_mae:.3f} eV/atom[/bold yellow]\n")
        
    except Exception as e:
        console.print(f"[red]Evaluation Error: {e}[/red]")

@app.command()
def substitute(formula: str, elem_out: str, elem_in: str):
    """
    (GNoME-inspired) Perform hypothetical elemental substitution to predict stability of a new material.
    e.g., matgraph substitute LiFePO4 Li Na
    """
    from matgraph.core import substitute_material
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        console.print("[red]Error: MP_API_KEY is not set.[/red]")
        raise typer.Exit(code=1)
        
    console.print(f"[cyan]Generative Discovery: Substituting {elem_out} with {elem_in} in {formula}...[/cyan]")
    try:
        res = substitute_material(formula, elem_out, elem_in, api_key)
        
        table = Table(title=f"Thermodynamic Stability Analysis")
        table.add_column("Material", style="magenta")
        table.add_column("Predicted Energy (M3GNet)", style="blue")
        table.add_column("Status", style="bold")
        
        o_energy = res['original']['energy']
        h_energy = res['hypothetical']['energy']
        
        table.add_row(res['original']['formula'] + " (Baseline)", f"{o_energy:.3f} eV", "[green]Known[/green]")
        
        status_color = "[bold green]Stable Candidate[/bold green]" if res['is_more_stable'] else "[bold red]Likely Unstable[/bold red]"
        table.add_row(res['hypothetical']['formula'] + " (Generated)", f"{h_energy:.3f} eV", status_color)
        
        console.print(table)
        
        if res['is_more_stable']:
            console.print(f"\n🎉 [bold green]Discovery![/bold green] The hypothetical material {res['hypothetical']['formula']} is predicted to be thermodynamically MORE stable than the baseline!")
        else:
            console.print(f"\n[yellow]Analysis:[/yellow] The substitution increases internal energy. {res['hypothetical']['formula']} might decompose.")
            
    except Exception as e:
        console.print(f"[red]Substitution Error: {e}[/red]")

@app.command()
def serve(port: int = 8000):
    """Start the robust GraphQL API server."""
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        console.print("[yellow]Warning: MP_API_KEY is not set. GraphQL queries will fail.[/yellow]")
        
    console.print(f"[green]Starting modern GraphQL server on port {port}...[/green]")
    console.print(f"[cyan]Explore the API at http://localhost:{port}/graphql[/cyan]")
    uvicorn.run("matgraph.graphql_app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    app()
