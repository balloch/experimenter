"""
Async tool functions called by both the MCP server and the CLI.

Each function returns a JSON string so results are easy to pass between tools.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

# Top-level imports so mock.patch("experimenter.tools.definitions.X") works.
from experimenter.planner import plan_experiment  # noqa: F401
from experimenter.data.discovery import search_all  # noqa: F401
from experimenter.state import load_run  # noqa: F401

logger = logging.getLogger(__name__)


# ── plan_experiment ────────────────────────────────────────────────────────────

async def tool_plan_experiment(
    question: str,
    time_budget_minutes: int = 30,
    compute: str = "auto",
) -> str:
    """
    Use Claude to design an ML experiment plan for the given research question.

    Returns JSON with the full ExperimentPlan including hypothesis, dataset query,
    recommended model type, compute backend, and cost estimate.
    """
    from experimenter.cloud.cost_estimator import estimate_cost

    plan = plan_experiment(
        question=question,
        time_budget_minutes=time_budget_minutes,
        compute=compute,
    )
    cost = estimate_cost(plan, compute=plan.compute)
    plan.estimated_cost_usd = cost.estimated_cost_usd

    return plan.model_dump_json(indent=2)


# ── search_datasets ────────────────────────────────────────────────────────────

async def tool_search_datasets(
    query: str,
    sources: Optional[list[str]] = None,
    limit_per_source: int = 5,
) -> str:
    """
    Search Kaggle, HuggingFace Hub, and the web for datasets matching the query.

    Returns a JSON array of DatasetCandidate objects sorted by relevance.
    """
    candidates = await search_all(query, sources=sources, limit_per_source=limit_per_source)
    return json.dumps([c.model_dump() for c in candidates], indent=2)


# ── download_dataset ───────────────────────────────────────────────────────────

def tool_download_dataset(
    dataset_id: str,
    source: str,
    url: str = "",
    name: str = "",
) -> str:
    """
    Download a dataset to the local cache.

    Returns JSON with {success, local_path, cached, error}.
    """
    from experimenter.data.loader import download_dataset
    from experimenter.models import DatasetCandidate

    candidate = DatasetCandidate(
        id=dataset_id,
        name=name or dataset_id,
        source=source,  # type: ignore[arg-type]
        description="",
        url=url or dataset_id,
    )
    result = download_dataset(candidate)
    return json.dumps(
        {
            "success": result.success,
            "local_path": result.local_path,
            "cached": result.cached,
            "error": result.error,
        }
    )


# ── prepare_data ───────────────────────────────────────────────────────────────

def tool_prepare_data(
    dataset_path: str,
    target_column: str,
    task_type: str,
    feature_columns: Optional[list[str]] = None,
    output_dir: Optional[str] = None,
) -> str:
    """
    Clean, impute, encode, and split a CSV dataset into train/test sets.

    Returns JSON with {prepared_path, train_path, test_path, profile}.
    """
    from experimenter.data.preparation import prepare_data

    try:
        result = prepare_data(
            dataset_path=dataset_path,
            target_column=target_column,
            task_type=task_type,
            feature_columns=feature_columns,
            output_dir=output_dir,
        )
        return json.dumps(
            {
                "prepared_path": result.prepared_path,
                "train_path": result.train_path,
                "test_path": result.test_path,
                "profile": result.profile.model_dump(),
            },
            indent=2,
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})


# ── estimate_cost ──────────────────────────────────────────────────────────────

def tool_estimate_cost(plan_json: str, compute: str = "auto") -> str:
    """
    Estimate the cost of running an experiment on the given compute backend.

    Returns JSON with {compute_backend, machine_type, estimated_cost_usd, recommendation}.
    """
    from experimenter.cloud.cost_estimator import estimate_cost
    from experimenter.models import ExperimentPlan

    plan = ExperimentPlan.model_validate_json(plan_json)
    estimate = estimate_cost(plan, compute=compute)
    return estimate.model_dump_json(indent=2)


# ── run_experiment ─────────────────────────────────────────────────────────────

def tool_run_experiment(
    plan_json: str,
    dataset_path: str,
    compute: str = "auto",
) -> str:
    """
    Start an ML training job (local or Vertex AI).

    Returns JSON with {experiment_id, status, compute_backend}.
    """
    from experimenter.models import ExperimentPlan, ExperimentRun
    from experimenter.state import save_run
    from experimenter.config import settings
    from experimenter.cloud.cost_estimator import estimate_cost

    plan = ExperimentPlan.model_validate_json(plan_json)
    cost_est = estimate_cost(plan, compute=compute)
    backend = cost_est.compute_backend

    run = ExperimentRun(plan=plan)
    run.status = "training"
    run.compute_backend = backend  # type: ignore[assignment]
    run.raw_dataset_path = dataset_path
    settings.ensure_dirs()
    save_run(run)

    if backend == "local":
        # Run synchronously in a thread so the MCP call returns quickly
        import threading

        def _train():
            from experimenter.cloud.local_runner import run_locally
            from experimenter.state import update_run_status
            from pathlib import Path

            out_dir = str(Path(run.run_dir) / "output")
            script = (
                "nlp"
                if plan.task_type in ("nlp_classification", "nlp_regression")
                else "tabular"
            )
            result = run_locally(
                script=script,
                data_path=dataset_path,
                target_column=plan.target_variable,
                task_type=plan.task_type,
                model_type=plan.model_type if plan.model_type != "auto" else "xgboost",
                output_dir=out_dir,
                wandb_disabled=not settings.has_wandb,
                wandb_project=settings.wandb_project,
                wandb_run_name=f"exp-{run.id[:8]}",
            )
            status = "complete" if result.success else "failed"
            update_run_status(
                run.id,
                status=status,
                metrics=result.metrics,
                error=result.error,
            )

        t = threading.Thread(target=_train, daemon=True)
        t.start()

    elif backend == "vertex_ai":
        from experimenter.cloud.vertex_ai import submit_vertex_job
        from experimenter.cloud.gcs import upload_file, dataset_gcs_uri, output_gcs_uri
        from experimenter.state import update_run_status

        try:
            data_gcs = dataset_gcs_uri(run.id)
            upload_file(dataset_path, data_gcs)
            job_name = submit_vertex_job(
                plan=plan,
                data_gcs_uri=data_gcs,
                output_gcs_uri=output_gcs_uri(run.id),
                wandb_api_key=settings.wandb_api_key,
                wandb_project=settings.wandb_project,
            )
            update_run_status(run.id, status="training")
            run.job_id = job_name
            save_run(run)
        except Exception as exc:
            update_run_status(run.id, status="failed", error=str(exc))

    return json.dumps(
        {
            "experiment_id": run.id,
            "status": "training",
            "compute_backend": backend,
        }
    )


# ── get_experiment_status ──────────────────────────────────────────────────────

def tool_get_experiment_status(experiment_id: str) -> str:
    """
    Get the current status of an experiment run.

    Returns JSON with {id, status, metrics, error} or {status: "not_found"}.
    """
    from experimenter.config import settings

    run = load_run(experiment_id)
    if run is None:
        return json.dumps({"status": "not_found", "id": experiment_id})

    # For Vertex AI jobs, refresh status from GCP
    if run.compute_backend == "vertex_ai" and run.job_id and run.status == "training":
        from experimenter.cloud.vertex_ai import get_job_status
        from experimenter.state import update_run_status

        gcp_status = get_job_status(run.job_id)
        if "succeeded" in gcp_status:
            update_run_status(experiment_id, status="complete")
            run.status = "complete"
        elif "failed" in gcp_status or "cancelled" in gcp_status:
            update_run_status(experiment_id, status="failed", error=gcp_status)
            run.status = "failed"

    return json.dumps(
        {
            "id": run.id,
            "status": run.status,
            "metrics": run.metrics,
            "error": run.error,
            "wandb_url": run.wandb_run_url,
        }
    )


# ── get_experiment_results ─────────────────────────────────────────────────────

def tool_get_experiment_results(experiment_id: str) -> str:
    """
    Retrieve full results for a completed experiment.

    Returns JSON with metrics, artifacts, W&B URL, and plan details.
    """
    from experimenter.state import load_run

    run = load_run(experiment_id)
    if run is None:
        return json.dumps({"error": f"Experiment {experiment_id} not found"})

    return json.dumps(
        {
            "id": run.id,
            "status": run.status,
            "metrics": run.metrics,
            "artifacts": run.artifacts,
            "wandb_url": run.wandb_run_url,
            "plan": run.plan.model_dump(),
        },
        indent=2,
    )


# ── answer_hypothesis ──────────────────────────────────────────────────────────

def tool_answer_hypothesis(
    experiment_id: str,
    question: str = "",
    hypothesis: str = "",
) -> str:
    """
    Analyze experiment results and produce a final natural-language answer.

    Returns JSON with {verdict, confidence, evidence_summary, key_metrics,
    feature_importance, limitations, full_report}.
    """
    from pathlib import Path
    from experimenter.state import load_run, save_run
    from experimenter.analysis.evaluator import evaluate_results
    from experimenter.analysis.reporter import generate_report

    run = load_run(experiment_id)
    if run is None:
        return json.dumps({"error": f"Experiment {experiment_id} not found"})

    if run.status not in ("complete", "failed"):
        return json.dumps(
            {"error": f"Experiment is still {run.status}. Wait for completion."}
        )

    # Find metrics.json
    metrics_path = Path(run.run_dir) / "output" / "metrics.json"
    if not metrics_path.exists():
        return json.dumps({"error": "metrics.json not found. Training may not have completed."})

    q = question or run.plan.question
    h = hypothesis or run.plan.hypothesis

    answer = evaluate_results(
        metrics_path=str(metrics_path),
        task_type=run.plan.task_type,
        hypothesis=h,
        target_variable=run.plan.target_variable,
        primary_feature=(run.plan.feature_variables[0] if run.plan.feature_variables else ""),
        question=q,
    )
    answer.wandb_url = run.wandb_run_url
    answer.full_report = generate_report(run, answer)

    run.answer = answer
    save_run(run)

    result = answer.model_dump()
    return json.dumps(result, indent=2)
