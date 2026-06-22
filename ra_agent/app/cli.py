"""CLI for MNO Revenue Assurance (agentic RAG)."""
from __future__ import annotations

import typer
from rich import print
from rich.console import Console
from rich.panel import Panel

from ra_agent.agent.agent import Agent
from ra_agent.agent.tools import SQLTool
from ra_agent.data import generate as datagen
from ra_agent.pipeline import build_warehouse, index_docs

app = typer.Typer(add_completion=False, help="MNO Revenue Assurance — agentic RAG pipeline")
console = Console()


@app.command()
def generate(cdrs: int = 30000, seed: int = 42):
    """Generate raw CDRs, billing extract, and RA docs."""
    res = datagen.generate_all(n_cdrs=cdrs, seed=seed)
    print(f"[green]Generated[/green] {res['cdrs']} CDRs, {res['billing_rows']} billing rows, "
          f"{res['enterprises']} enterprises, {res['docs']} RA docs")


@app.command()
def warehouse():
    """Build the DuckDB star schema (+ Parquet export)."""
    counts = build_warehouse.build()
    print(f"[green]Warehouse built:[/green] {counts}")


@app.command()
def index():
    """Index RA methodology docs into the vector store."""
    res = index_docs.index_docs()
    print(f"[green]Indexed[/green] {res['indexed']} chunks (total {res['total']})")


@app.command()
def pipeline(cdrs: int = 30000, seed: int = 42):
    """Run the full pipeline: generate -> warehouse -> index."""
    res = datagen.generate_all(n_cdrs=cdrs, seed=seed)
    print(f"[green]Data:[/green] {res['cdrs']} CDRs, {res['docs']} docs")
    counts = build_warehouse.build()
    print(f"[green]Warehouse:[/green] {counts}")
    ires = index_docs.index_docs()
    print(f"[green]Indexed[/green] {ires['indexed']} doc chunks")
    print("[bold green]Pipeline complete.[/bold green] Try: investigate \"...\"")


@app.command()
def query(sql: str):
    """Run a read-only SQL query against the warehouse."""
    print(SQLTool().run(sql).observation)


@app.command()
def investigate(question: str = typer.Argument("Estimate total A2P OTP revenue leakage and recommend controls.")):
    """Run the agent on an investigation question."""
    res = Agent().run(question)
    console.print(Panel.fit(f"[bold]Agent backend:[/bold] {res.backend}  |  steps: {len(res.steps)}"))
    for i, s in enumerate(res.steps, 1):
        console.print(f"[dim]Step {i}[/dim] [yellow]{s.action}[/yellow]({s.action_input})")
        console.print(f"   [dim]thought:[/dim] {s.thought}")
        obs = s.observation if len(s.observation) < 400 else s.observation[:400] + "..."
        console.print(f"   [dim]obs:[/dim] {obs}")
    console.print(Panel(res.answer, title="Investigation Report"))
    print(f"[dim]Sources: {', '.join(res.sources)}[/dim]")


def main():
    app()


if __name__ == "__main__":
    main()
