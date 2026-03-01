"""GCP Vertex AI machine cost estimation."""

from __future__ import annotations

from experimenter.models import CostEstimate, ExperimentPlan

# ── Vertex AI hourly rates (USD, preemptible/spot) ────────────────────────────
# Source: GCP pricing as of early 2026 (us-central1, spot instances)
MACHINE_HOURLY_RATES: dict[str, float] = {
    "n1-standard-2": 0.019,
    "n1-standard-4": 0.038,
    "n1-standard-8": 0.076,
    "n1-standard-16": 0.152,
    "n1-standard-32": 0.304,
    "n1-highmem-4": 0.050,
    "n1-highmem-8": 0.100,
    "n2-standard-4": 0.042,
    "n2-standard-8": 0.084,
    # GPU machines (base + accelerator)
    "n1-standard-4-t4": 0.11,   # + NVIDIA T4 spot ~$0.072/hr
    "n1-standard-8-t4": 0.148,
    "a2-highgpu-1g": 0.75,      # A100 GPU
}

# Minimum time budget (minutes) to prefer Vertex AI over local
VERTEX_MIN_MINUTES = 10


def recommend_machine(task_type: str, needs_gpu: bool, time_budget_minutes: int) -> str:
    """Return the most cost-effective Vertex AI machine type for the task."""
    if needs_gpu or task_type in ("nlp_classification", "nlp_regression"):
        return "n1-standard-4-t4"
    if time_budget_minutes <= 15:
        return "n1-standard-4"
    if time_budget_minutes <= 60:
        return "n1-standard-8"
    return "n1-standard-16"


def estimate_cost(plan: ExperimentPlan, compute: str) -> CostEstimate:
    """
    Estimate the cost of running the experiment.

    compute: "local" | "vertex_ai" | "auto"
    """
    if compute == "auto":
        # Use local for very short budgets or when GCP isn't configured
        from experimenter.config import settings

        if plan.time_budget_minutes < VERTEX_MIN_MINUTES or not settings.has_gcp:
            compute = "local"
        else:
            compute = "vertex_ai"

    if compute == "local":
        return CostEstimate(
            compute_backend="local",
            machine_type="local",
            estimated_hours=plan.time_budget_minutes / 60,
            estimated_cost_usd=0.0,
            recommendation=(
                "Running locally — free but limited to your machine's CPU/RAM. "
                "Consider Vertex AI for larger datasets or longer training runs."
            ),
        )

    # Vertex AI
    machine_type = plan.machine_type
    if machine_type not in MACHINE_HOURLY_RATES:
        machine_type = recommend_machine(
            plan.task_type, plan.needs_gpu, plan.time_budget_minutes
        )

    hourly_rate = MACHINE_HOURLY_RATES[machine_type]
    hours = plan.time_budget_minutes / 60
    cost = hourly_rate * hours

    return CostEstimate(
        compute_backend="vertex_ai",
        machine_type=machine_type,
        estimated_hours=hours,
        estimated_cost_usd=round(cost, 4),
        breakdown={
            "machine_hourly_rate_usd": hourly_rate,
            "hours": hours,
        },
        recommendation=(
            f"Vertex AI {machine_type} spot instance (~${hourly_rate:.3f}/hr). "
            f"Estimated cost: ${cost:.3f} for {plan.time_budget_minutes} minutes. "
            "Uses preemptible/spot pricing for maximum savings."
        ),
    )
