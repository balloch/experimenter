"""Evaluate experiment results and render a hypothesis verdict."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experimenter.models import HypothesisAnswer

logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────
R2_CONFIRMED = 0.40       # R² ≥ 0.40 → hypothesis confirmed
R2_INCONCLUSIVE = 0.10   # 0.10 ≤ R² < 0.40 → inconclusive
# R² < 0.10 → rejected

ACCURACY_LIFT_CONFIRMED = 0.15    # accuracy ≥ baseline + 15pp → confirmed
ACCURACY_LIFT_INCONCLUSIVE = 0.05  # baseline + 5pp → inconclusive


class StatisticalTest:
    @staticmethod
    def regression_verdict(r2: float) -> str:
        if r2 >= R2_CONFIRMED:
            return "confirmed"
        if r2 >= R2_INCONCLUSIVE:
            return "inconclusive"
        return "rejected"

    @staticmethod
    def classification_verdict(accuracy: float, baseline: float = 0.5) -> str:
        lift = accuracy - baseline
        if lift >= ACCURACY_LIFT_CONFIRMED:
            return "confirmed"
        if lift >= ACCURACY_LIFT_INCONCLUSIVE:
            return "inconclusive"
        return "rejected"


def evaluate_results(
    metrics_path: str,
    task_type: str,
    hypothesis: str,
    target_variable: str,
    primary_feature: str,
    question: str = "",
) -> HypothesisAnswer:
    """
    Load metrics from a JSON file and produce a HypothesisAnswer.
    """
    metrics: dict[str, Any] = json.loads(Path(metrics_path).read_text())

    verdict, confidence, key_metrics, limitations = _assess(metrics, task_type)

    feature_importance: dict[str, float] = {}
    if "feature_importance" in metrics:
        fi_raw = metrics["feature_importance"]
        # Sort descending by importance
        feature_importance = dict(
            sorted(fi_raw.items(), key=lambda x: x[1], reverse=True)
        )

    return HypothesisAnswer(
        question=question,
        hypothesis=hypothesis,
        verdict=verdict,
        confidence=confidence,
        evidence_summary=_build_summary(verdict, confidence, metrics, task_type, primary_feature),
        key_metrics=key_metrics,
        feature_importance=feature_importance,
        limitations=limitations,
    )


# ── Private ────────────────────────────────────────────────────────────────────

def _assess(
    metrics: dict[str, Any], task_type: str
) -> tuple[str, float, dict[str, float], list[str]]:
    limitations: list[str] = []
    key_metrics: dict[str, float] = {}

    if task_type in ("regression", "time_series"):
        r2 = float(metrics.get("r2", 0.0))
        rmse = float(metrics.get("rmse", 0.0))
        key_metrics = {"r2": r2, "rmse": rmse}
        verdict = StatisticalTest.regression_verdict(r2)
        confidence = _r2_to_confidence(r2)

        if r2 < R2_CONFIRMED:
            limitations.append(
                f"Model explains only {r2 * 100:.1f}% of variance (R²={r2:.3f}). "
                "Other factors not captured in the dataset may be important."
            )

    elif task_type in ("binary_classification", "multiclass_classification"):
        accuracy = float(metrics.get("accuracy", 0.5))
        f1 = float(metrics.get("f1", 0.0))
        key_metrics = {"accuracy": accuracy, "f1": f1}

        n_classes = metrics.get("n_classes", 2)
        baseline = 1.0 / max(int(n_classes), 2)
        verdict = StatisticalTest.classification_verdict(accuracy, baseline)
        confidence = _accuracy_to_confidence(accuracy, baseline)

        if accuracy < baseline + ACCURACY_LIFT_CONFIRMED:
            limitations.append(
                f"Model accuracy ({accuracy:.1%}) is close to the baseline ({baseline:.1%}). "
                "The features may have limited predictive power."
            )

    else:
        # Fallback: treat any metric > 0.5 as supporting
        primary = list(metrics.values())[0] if metrics else 0.0
        verdict = "inconclusive"
        confidence = 0.5
        key_metrics = metrics

    if not metrics:
        limitations.append("No metrics were produced — training may have failed silently.")

    return verdict, confidence, key_metrics, limitations


def _r2_to_confidence(r2: float) -> float:
    """Map R² (−∞ to 1) to confidence (0 to 1)."""
    r2 = max(0.0, min(1.0, r2))
    return round(min(0.99, r2 + 0.1), 3)


def _accuracy_to_confidence(accuracy: float, baseline: float) -> float:
    """Map accuracy lift above baseline to confidence."""
    lift = accuracy - baseline
    confidence = min(0.99, max(0.01, lift / (1.0 - baseline)))
    return round(confidence, 3)


def _build_summary(
    verdict: str,
    confidence: float,
    metrics: dict,
    task_type: str,
    primary_feature: str,
) -> str:
    pct = f"{confidence * 100:.0f}%"
    if task_type in ("regression", "time_series"):
        r2 = metrics.get("r2", 0)
        return (
            f"The model explains {float(r2) * 100:.1f}% of the variance in the target "
            f"(R²={float(r2):.3f}), suggesting the hypothesis is {verdict} "
            f"with {pct} confidence. "
            f"'{primary_feature}' is among the features considered."
        )
    else:
        acc = metrics.get("accuracy", 0)
        return (
            f"The classifier achieved {float(acc) * 100:.1f}% accuracy, "
            f"indicating the hypothesis is {verdict} with {pct} confidence."
        )
