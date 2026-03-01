"""Experiment planner — uses Claude to turn a question into an ExperimentPlan."""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from anthropic import Anthropic

from experimenter.models import ExperimentPlan

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert ML experiment designer.
Given a scientific question, output a JSON object describing the ML experiment needed to answer it.

The JSON must contain exactly these keys:
- "hypothesis": string — a specific, testable hypothesis
- "target_variable": string — the column name to predict (snake_case)
- "feature_variables": list[string] — column names to use as features
- "task_type": one of "binary_classification" | "multiclass_classification" | "regression" | "nlp_classification" | "nlp_regression" | "clustering" | "time_series"
- "dataset_query": string — a web/Kaggle/HuggingFace search query to find relevant data
- "model_type": one of "xgboost" | "lightgbm" | "random_forest" | "logistic_regression" | "linear_regression" | "bert" | "distilbert" | "auto"
- "needs_gpu": boolean — true only for deep learning tasks
- "machine_type": string — Vertex AI machine type, e.g. "n1-standard-4"
- "description": string — 1-2 sentence human-readable plan summary

Output ONLY valid JSON. Do not include markdown fences or explanatory text."""


def plan_experiment(
    question: str,
    time_budget_minutes: int = 30,
    compute: str = "auto",
    anthropic_api_key: Optional[str] = None,
) -> ExperimentPlan:
    """
    Use Claude to design an ML experiment plan for the given question.

    Returns an ExperimentPlan ready to execute.
    Raises ValueError if the response cannot be parsed.
    """
    from experimenter.config import settings

    api_key = anthropic_api_key or settings.anthropic_api_key
    client = Anthropic(api_key=api_key)

    user_message = (
        f"Design an ML experiment to answer this question:\n\n"
        f"{question}\n\n"
        f"Time budget: {time_budget_minutes} minutes.\n"
        f"Compute preference: {compute}."
    )

    response = client.messages.create(
        model=settings.planner_model,
        max_tokens=1000,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw_text = response.content[0].text
    plan_data = _parse_plan_from_llm_response(raw_text)

    return ExperimentPlan(
        question=question,
        time_budget_minutes=time_budget_minutes,
        compute=compute,  # type: ignore[arg-type]
        **plan_data,
    )


def _parse_plan_from_llm_response(raw: str) -> dict:
    """
    Extract and parse JSON from the LLM response.
    Handles plain JSON or JSON wrapped in ``` or ```json fences.
    Raises ValueError if no valid JSON is found.
    """
    # Try plain JSON first
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Try code fence extraction
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw, re.IGNORECASE)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding any JSON object in the text
    brace_match = re.search(r"\{[\s\S]+\}", raw)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not extract a valid experiment plan from the LLM response. "
        f"Raw response (first 500 chars): {raw[:500]}"
    )
