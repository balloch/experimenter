"""
Experimenter CLI.

Usage:
  experimenter plan "Does exercise predict GPA?"
  experimenter search "student exercise GPA"
  experimenter run "Does income predict satisfaction?" --compute local --time-budget 10
  experimenter status <experiment-id>
  experimenter results <experiment-id>
  experimenter list
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

import typer
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(
    name="experimenter",
    help="ML experiment platform orchestrated by Claude.",
    rich_markup_mode="rich",
)
console = Console()


# ── plan ───────────────────────────────────────────────────────────────────────

@app.command()
def plan(
    question: str = typer.Argument(..., help="Research question to answer"),
    time_budget: int = typer.Option(30, "--time-budget", "-t", help="Time budget in minutes"),
    compute: str = typer.Option("auto", "--compute", "-c", help="local | vertex_ai | auto"),
):
    """Design an experiment plan without running it (plan mode)."""
    from experimenter.tools.definitions import tool_plan_experiment

    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}")) as p:
        p.add_task("Asking Claude to design the experiment...")
        result = asyncio.run(
            tool_plan_experiment(question=question, time_budget_minutes=time_budget, compute=compute)
        )

    data = json.loads(result)
    console.print(Panel(JSON(result), title="[bold green]Experiment Plan", expand=False))
    console.print(f"\n[bold]Plan ID:[/bold] {data['id']}")
    console.print(f"[bold]Hypothesis:[/bold] {data['hypothesis']}")
    console.print(f"[bold]Model:[/bold] {data['model_type']}  |  [bold]Compute:[/bold] {data['compute']}")
    console.print(f"[bold]Estimated cost:[/bold] ${data.get('estimated_cost_usd', 0):.4f}")


# ── search ─────────────────────────────────────────────────────────────────────

@app.command()
def search(
    query: str = typer.Argument(..., help="Dataset search query"),
    sources: Optional[str] = typer.Option(None, help="Comma-separated: kaggle,huggingface,web"),
    limit: int = typer.Option(5, help="Results per source"),
):
    """Search for datasets on Kaggle, HuggingFace, and the web."""
    from experimenter.tools.definitions import tool_search_datasets

    src_list = [s.strip() for s in sources.split(",")] if sources else None
    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}")) as p:
        p.add_task(f"Searching {src_list or 'all sources'}...")
        result = asyncio.run(tool_search_datasets(query=query, sources=src_list, limit_per_source=limit))

    candidates = json.loads(result)
    table = Table(title=f"Datasets for: {query!r}")
    table.add_column("Source", style="cyan")
    table.add_column("ID", style="yellow")
    table.add_column("Name")
    table.add_column("Score", justify="right")
    for c in candidates:
        table.add_row(c["source"], c["id"], c["name"][:60], f"{c['relevance_score']:.2f}")
    console.print(table)


# ── run (end-to-end) ───────────────────────────────────────────────────────────

@app.command()
def run(
    question: str = typer.Argument(..., help="Research question to answer"),
    time_budget: int = typer.Option(30, "--time-budget", "-t"),
    compute: str = typer.Option("auto", "--compute", "-c"),
    dataset_id: Optional[str] = typer.Option(None, "--dataset", help="Pre-selected dataset ID"),
    dataset_source: str = typer.Option("kaggle", "--source"),
    dataset_url: str = typer.Option("", "--url"),
    poll_interval: int = typer.Option(10, "--poll", help="Status poll interval (seconds)"),
):
    """Run a full end-to-end experiment to answer a research question."""
    from experimenter.tools.definitions import (
        tool_plan_experiment,
        tool_search_datasets,
        tool_download_dataset,
        tool_prepare_data,
        tool_run_experiment,
        tool_get_experiment_status,
        tool_answer_hypothesis,
    )

    console.rule("[bold blue]Step 1/6 — Planning experiment")
    plan_json = asyncio.run(
        tool_plan_experiment(question=question, time_budget_minutes=time_budget, compute=compute)
    )
    plan_data = json.loads(plan_json)
    console.print(f"[green]✓[/green] Hypothesis: {plan_data['hypothesis']}")
    console.print(f"[green]✓[/green] Model: {plan_data['model_type']} | Dataset query: {plan_data['dataset_query']!r}")

    console.rule("[bold blue]Step 2/6 — Finding datasets")
    if not dataset_id:
        candidates_json = asyncio.run(tool_search_datasets(query=plan_data["dataset_query"]))
        candidates = json.loads(candidates_json)
        if not candidates:
            console.print("[red]No datasets found. Try --dataset to specify one manually.[/red]")
            raise typer.Exit(1)
        best = candidates[0]
        dataset_id = best["id"]
        dataset_source = best["source"]
        dataset_url = best["url"]
        console.print(f"[green]✓[/green] Using: {best['name']} ({best['source']})")

    console.rule("[bold blue]Step 3/6 — Downloading dataset")
    dl_json = tool_download_dataset(dataset_id=dataset_id, source=dataset_source, url=dataset_url)
    dl = json.loads(dl_json)
    if not dl["success"]:
        console.print(f"[red]Download failed: {dl['error']}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]✓[/green] {dl['local_path']} {'(cached)' if dl['cached'] else ''}")

    console.rule("[bold blue]Step 4/6 — Preparing data")
    prep_json = tool_prepare_data(
        dataset_path=dl["local_path"],
        target_column=plan_data["target_variable"],
        task_type=plan_data["task_type"],
    )
    prep = json.loads(prep_json)
    if "error" in prep:
        console.print(f"[red]Prep failed: {prep['error']}[/red]")
        raise typer.Exit(1)
    profile = prep["profile"]
    console.print(f"[green]✓[/green] {profile['num_rows']} rows × {profile['num_columns']} cols")
    if profile.get("warnings"):
        for w in profile["warnings"]:
            console.print(f"  [yellow]⚠[/yellow] {w}")

    console.rule("[bold blue]Step 5/6 — Training")
    run_json = tool_run_experiment(
        plan_json=plan_json, dataset_path=prep["prepared_path"], compute=compute
    )
    run_data = json.loads(run_json)
    experiment_id = run_data["experiment_id"]
    console.print(f"[green]✓[/green] Job started on {run_data['compute_backend']} | ID: {experiment_id}")

    # Poll until complete
    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}")) as p:
        task = p.add_task("Waiting for training to complete...")
        while True:
            status_json = tool_get_experiment_status(experiment_id)
            status = json.loads(status_json)
            p.update(task, description=f"Status: {status['status']} ...")
            if status["status"] in ("complete", "failed"):
                break
            time.sleep(poll_interval)

    if status["status"] == "failed":
        console.print(f"[red]Training failed: {status.get('error')}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]✓[/green] Metrics: {status.get('metrics', {})}")

    console.rule("[bold blue]Step 6/6 — Answering hypothesis")
    answer_json = tool_answer_hypothesis(experiment_id=experiment_id)
    answer = json.loads(answer_json)

    verdict_color = {"confirmed": "green", "rejected": "red", "inconclusive": "yellow"}.get(
        answer.get("verdict", ""), "white"
    )
    console.print(
        Panel(
            f"[bold {verdict_color}]{answer.get('verdict', '').upper()}[/bold {verdict_color}] "
            f"(confidence: {answer.get('confidence', 0):.0%})\n\n"
            f"{answer.get('evidence_summary', '')}",
            title=f"[bold]Answer: {question}[/bold]",
        )
    )
    if answer.get("full_report"):
        console.print(answer["full_report"])


# ── status ─────────────────────────────────────────────────────────────────────

@app.command()
def status(experiment_id: str = typer.Argument(...)):
    """Check the status of an experiment."""
    from experimenter.tools.definitions import tool_get_experiment_status

    result = tool_get_experiment_status(experiment_id)
    console.print(JSON(result))


# ── results ────────────────────────────────────────────────────────────────────

@app.command()
def results(experiment_id: str = typer.Argument(...)):
    """Show full results of a completed experiment."""
    from experimenter.tools.definitions import tool_get_experiment_results

    result = tool_get_experiment_results(experiment_id)
    console.print(JSON(result))


# ── list ───────────────────────────────────────────────────────────────────────

@app.command(name="list")
def list_experiments():
    """List all experiment runs."""
    from experimenter.state import list_runs

    runs = list_runs()
    if not runs:
        console.print("[dim]No experiments yet. Run: experimenter run \"your question\"[/dim]")
        return

    table = Table(title="Experiments")
    table.add_column("ID", style="dim")
    table.add_column("Question")
    table.add_column("Status", style="cyan")
    table.add_column("Model")
    table.add_column("Verdict")
    for r in runs:
        verdict = r.answer.verdict if r.answer else "-"
        table.add_row(
            r.id[:12] + "...",
            r.plan.question[:50],
            r.status,
            r.plan.model_type,
            verdict,
        )
    console.print(table)


if __name__ == "__main__":
    app()
