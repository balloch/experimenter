"""
MCP server for the Experimenter platform.

Run with:  experimenter-mcp
       or: python -m experimenter.mcp_server

Add to Claude Desktop's MCP config:
  {
    "mcpServers": {
      "experimenter": {
        "command": "experimenter-mcp",
        "env": { "ANTHROPIC_API_KEY": "..." }
      }
    }
  }
"""

from __future__ import annotations

import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from experimenter.tools.definitions import (
    tool_answer_hypothesis,
    tool_download_dataset,
    tool_estimate_cost,
    tool_get_experiment_results,
    tool_get_experiment_status,
    tool_plan_experiment,
    tool_prepare_data,
    tool_run_experiment,
    tool_search_datasets,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "experimenter",
    instructions=(
        "ML Experiment Platform. Use these tools to answer scientific questions by "
        "running real ML experiments. Typical flow:\n"
        "1. plan_experiment — design the experiment\n"
        "2. search_datasets — find relevant data\n"
        "3. download_dataset — download best candidate\n"
        "4. prepare_data — clean & split\n"
        "5. estimate_cost — choose compute backend\n"
        "6. run_experiment — train the model\n"
        "7. get_experiment_status — poll until complete\n"
        "8. answer_hypothesis — interpret results and answer the question"
    ),
)


@mcp.tool()
async def plan_experiment(
    question: str,
    time_budget_minutes: int = 30,
    compute: str = "auto",
) -> str:
    """
    Design an ML experiment to answer a research question.

    Returns a JSON ExperimentPlan with hypothesis, recommended model, dataset
    search query, compute backend, and cost estimate. Review this plan before
    executing — use it as the basis for all subsequent tool calls.
    """
    return await tool_plan_experiment(
        question=question,
        time_budget_minutes=time_budget_minutes,
        compute=compute,
    )


@mcp.tool()
async def search_datasets(
    query: str,
    sources: Optional[list[str]] = None,
    limit_per_source: int = 5,
) -> str:
    """
    Search Kaggle, HuggingFace, and the web for datasets matching the query.

    sources: list of "kaggle" | "huggingface" | "web" (default: all three)
    Returns a JSON array of DatasetCandidate objects sorted by relevance score.
    """
    return await tool_search_datasets(
        query=query, sources=sources, limit_per_source=limit_per_source
    )


@mcp.tool()
def download_dataset(
    dataset_id: str,
    source: str,
    url: str = "",
    name: str = "",
) -> str:
    """
    Download a dataset to the local cache.

    source: "kaggle" | "huggingface" | "url" | "web"
    dataset_id: Kaggle ref (owner/slug), HuggingFace dataset id, or URL.
    Returns {success, local_path, cached, error}.
    """
    return tool_download_dataset(
        dataset_id=dataset_id, source=source, url=url, name=name
    )


@mcp.tool()
def prepare_data(
    dataset_path: str,
    target_column: str,
    task_type: str,
    feature_columns: Optional[list[str]] = None,
    output_dir: Optional[str] = None,
) -> str:
    """
    Prepare a CSV dataset for ML: impute missing values, encode categoricals,
    and produce train/test splits.

    task_type: "regression" | "binary_classification" | "multiclass_classification"
               | "nlp_classification" | "time_series" | "clustering"
    Returns {prepared_path, train_path, test_path, profile}.
    """
    return tool_prepare_data(
        dataset_path=dataset_path,
        target_column=target_column,
        task_type=task_type,
        feature_columns=feature_columns,
        output_dir=output_dir,
    )


@mcp.tool()
def estimate_cost(plan_json: str, compute: str = "auto") -> str:
    """
    Estimate the cost of running an experiment.

    plan_json: JSON string from plan_experiment.
    compute: "local" | "vertex_ai" | "auto"
    Returns {compute_backend, machine_type, estimated_cost_usd, recommendation}.
    """
    return tool_estimate_cost(plan_json=plan_json, compute=compute)


@mcp.tool()
def run_experiment(
    plan_json: str,
    dataset_path: str,
    compute: str = "auto",
) -> str:
    """
    Start the ML training job (local subprocess or Vertex AI).

    plan_json: JSON string from plan_experiment.
    dataset_path: local path to the prepared training CSV.
    Returns {experiment_id, status, compute_backend}.

    The job runs asynchronously. Poll get_experiment_status until complete.
    """
    return tool_run_experiment(
        plan_json=plan_json, dataset_path=dataset_path, compute=compute
    )


@mcp.tool()
def get_experiment_status(experiment_id: str) -> str:
    """
    Check the current status of a running experiment.

    Returns {id, status, metrics, error, wandb_url}.
    status is one of: pending | searching | downloading | preparing |
                      training | analyzing | complete | failed
    """
    return tool_get_experiment_status(experiment_id=experiment_id)


@mcp.tool()
def get_experiment_results(experiment_id: str) -> str:
    """
    Retrieve the full results of a completed experiment.

    Returns {id, status, metrics, artifacts, wandb_url, plan}.
    """
    return tool_get_experiment_results(experiment_id=experiment_id)


@mcp.tool()
def answer_hypothesis(
    experiment_id: str,
    question: str = "",
    hypothesis: str = "",
) -> str:
    """
    Analyze experiment results and produce a final answer to the research question.

    Returns {verdict, confidence, evidence_summary, key_metrics,
    feature_importance, limitations, full_report} where verdict is one of
    "confirmed" | "rejected" | "inconclusive".
    """
    return tool_answer_hypothesis(
        experiment_id=experiment_id,
        question=question,
        hypothesis=hypothesis,
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
