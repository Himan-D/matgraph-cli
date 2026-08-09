import typer
from rich.console import Console
from rich.table import Table
import os
import uvicorn
from typing import Optional
from matgraph.core import run_pipeline, save_results

app = typer.Typer(help="🔬 MatGraph CLI: The complete Material Science ML Product.")
console = Console()

@app.command()
def setup(api_key: str):
    """Set up your Materials Project API Key."""
    console.print(f"[green]✅ Setup Instructions:[/green]")
    console.print("To use the CLI and API, export your API key in your terminal:")
    console.print(f'[bold cyan]export MP_API_KEY="{api_key}"[/bold cyan]')

@app.command()
def predict(
    formula: str, 
    min_gap: Optional[float] = typer.Option(None, "--min-gap", help="Minimum true band gap to filter (eV)"),
    max_gap: Optional[float] = typer.Option(None, "--max-gap", help="Maximum true band gap to filter (eV)"),
    crystal_system: Optional[str] = typer.Option(None, "--crystal-system", help="Filter by crystal system (e.g., Cubic, Hexagonal)"),
    save: Optional[str] = typer.Option(None, "--save", help="File path to save results (e.g., results.csv)"),
    format: str = typer.Option("csv", "--format", help="Save format: 'csv' or 'json'")
):
    """Run the complete ML pipeline with advanced search filters and data saving."""
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        console.print("[red]❌ Error: MP_API_KEY is not set.[/red]")
        raise typer.Exit(code=1)
    
    console.print(f"[cyan]🔍 Searching and running ML pipeline for {formula}...[/cyan]")
    if min_gap or max_gap or crystal_system:
        console.print(f"[dim]Filters applied - Min Gap: {min_gap}, Max Gap: {max_gap}, System: {crystal_system}[/dim]")
    
    try:
        results = run_pipeline(
            formula=formula, 
            api_key=api_key, 
            min_gap=min_gap, 
            max_gap=max_gap, 
            crystal_system=crystal_system
        )
        
        if not results:
            console.print(f"[yellow]⚠️ No materials found matching the criteria.[/yellow]")
            return
            
        table = Table(title=f"🧪 Prediction Results for {formula}")
        table.add_column("Material ID", style="cyan")
        table.add_column("Formula", style="magenta")
        table.add_column("System", style="blue")
        table.add_column("True Gap (eV)", style="green")
        table.add_column("Pred Gap (eV)", style="bold blue")
        table.add_column("Density", style="yellow")
        
        for r in results:
            table.add_row(
                str(r["material_id"]),
                r["formula"],
                str(r["crystal_system"]),
                str(r["true_band_gap"]),
                str(r["predicted_band_gap"]),
                f'{r["features"]["density"]:.2f}'
            )
            
        console.print(table)
        console.print(f"[green]✨ Pipeline completed successfully! Found {len(results)} items.[/green]")
        
        if save:
            save_results(results, save, format.lower())
            console.print(f"[bold green]💾 Saved {len(results)} results to {save} (Format: {format.upper()})[/bold green]")

    except Exception as e:
        console.print(f"[red]❌ Pipeline Error: {e}[/red]")

@app.command()
def serve(port: int = 8000):
    """Start the robust GraphQL API server."""
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        console.print("[yellow]⚠️ Warning: MP_API_KEY is not set. GraphQL queries will fail.[/yellow]")
        
    console.print(f"[green]🚀 Starting modern GraphQL server on port {port}...[/green]")
    console.print(f"[cyan]🌐 Explore the API at http://localhost:{port}/graphql[/cyan]")
    uvicorn.run("matgraph.graphql_app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    app()
