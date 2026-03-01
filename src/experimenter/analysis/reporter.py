"""Generate a natural-language research report using Claude."""

from __future__ import annotations

import logging
from typing import Optional

from experimenter.models import ExperimentRun, HypothesisAnswer

logger = logging.getLogger(__name__)


def generate_report(
    run: ExperimentRun,
    answer: HypothesisAnswer,
    model: Optional[str] = None,
) -> str:
    """
    Use Claude to write a concise, human-readable research report.
    Returns markdown text.
    Falls back to a template-based report if the API call fails.
    """
    from experimenter.config import settings

    if model is None:
        model = settings.reporter_model

    prompt = _build_prompt(run, answer)

    if not settings.anthropic_api_key:
        logger.warning("ANTHROPIC_API_KEY not set — using template report")
        return _template_report(run, answer)

    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model=model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as exc:
        logger.warning("Claude report generation failed (%s) — using template", exc)
        return _template_report(run, answer)


def _build_prompt(run: ExperimentRun, answer: HypothesisAnswer) -> str:
    fi_text = ""
    if answer.feature_importance:
        top = sorted(answer.feature_importance.items(), key=lambda x: -x[1])[:5]
        fi_text = "\n".join(f"  - {feat}: {imp:.3f}" for feat, imp in top)

    limitations_text = "\n".join(f"  - {l}" for l in answer.limitations)

    return f"""You are a data scientist writing a research summary. Based on the ML experiment below, write a concise markdown report (max 300 words) that:
1. Directly answers the original question
2. Summarizes the key evidence
3. States confidence level and any important caveats

---
**Question**: {run.plan.question}
**Hypothesis**: {run.plan.hypothesis}
**Verdict**: {answer.verdict.upper()} (confidence: {answer.confidence:.0%})
**Task type**: {run.plan.task_type}
**Model**: {run.plan.model_type}
**Key metrics**: {answer.key_metrics}
**Top feature importances**:
{fi_text or "  (not available)"}
**Limitations**:
{limitations_text or "  None identified"}
**W&B run**: {answer.wandb_url or "N/A"}
---

Write the report now:"""


def _template_report(run: ExperimentRun, answer: HypothesisAnswer) -> str:
    """Fallback template-based report when Claude is unavailable."""
    verdict_emoji = {"confirmed": "✅", "rejected": "❌", "inconclusive": "⚠️"}.get(
        answer.verdict, "?"
    )
    metrics_lines = "\n".join(
        f"- **{k}**: {v:.4f}" if isinstance(v, float) else f"- **{k}**: {v}"
        for k, v in answer.key_metrics.items()
    )
    fi_lines = ""
    if answer.feature_importance:
        top = sorted(answer.feature_importance.items(), key=lambda x: -x[1])[:5]
        fi_lines = "\n".join(f"- `{f}`: {i:.4f}" for f, i in top)

    limitations = "\n".join(f"- {l}" for l in answer.limitations) or "None identified."

    return f"""# Experiment Report

## Question
{run.plan.question}

## Verdict: {verdict_emoji} {answer.verdict.upper()}
*Confidence: {answer.confidence:.0%}*

## Summary
{answer.evidence_summary}

## Key Metrics
{metrics_lines}

{"## Feature Importance (top 5)" + chr(10) + fi_lines if fi_lines else ""}

## Limitations
{limitations}

{"## W&B Run" + chr(10) + answer.wandb_url if answer.wandb_url else ""}
"""
