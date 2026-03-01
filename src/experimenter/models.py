"""Pydantic data models shared across the platform."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Dataset discovery ──────────────────────────────────────────────────────────

class DatasetCandidate(BaseModel):
    id: str
    name: str
    source: Literal["kaggle", "huggingface", "url", "web"]
    description: str
    url: str
    size_mb: Optional[float] = None
    num_rows: Optional[int] = None
    num_columns: Optional[int] = None
    relevance_score: float = 0.0
    tags: list[str] = []


# ── Experiment planning ────────────────────────────────────────────────────────

class ExperimentPlan(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    hypothesis: str
    target_variable: str
    feature_variables: list[str] = []
    task_type: Literal[
        "binary_classification",
        "multiclass_classification",
        "regression",
        "nlp_classification",
        "nlp_regression",
        "clustering",
        "time_series",
    ]
    dataset_query: str                          # search query for dataset discovery
    suggested_dataset_ids: list[str] = []       # optional pre-selected dataset IDs

    # Model selection
    model_type: Literal[
        "xgboost", "lightgbm", "random_forest",
        "logistic_regression", "linear_regression",
        "bert", "distilbert", "auto",
    ] = "auto"
    needs_gpu: bool = False

    # Compute
    compute: Literal["local", "vertex_ai", "auto"] = "auto"
    machine_type: str = "n1-standard-4"         # Vertex AI machine type

    # Timing / cost
    time_budget_minutes: int = 30
    estimated_cost_usd: float = 0.0

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    description: str = ""                       # human-readable plan summary


# ── Data profiling ─────────────────────────────────────────────────────────────

class ColumnProfile(BaseModel):
    dtype: str
    null_pct: float
    unique_count: int
    sample_values: list[Any] = []


class DataProfile(BaseModel):
    num_rows: int
    num_columns: int
    columns: dict[str, ColumnProfile]
    target_column: str
    target_distribution: dict[str, Any]         # class counts or {mean, std, min, max}
    recommended_task_type: str
    warnings: list[str] = []


# ── Experiment runtime ─────────────────────────────────────────────────────────

class ExperimentRun(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    plan: ExperimentPlan

    # Data paths
    raw_dataset_path: Optional[str] = None
    prepared_data_path: Optional[str] = None
    data_profile: Optional[DataProfile] = None

    # Execution
    status: Literal[
        "pending", "searching", "downloading", "preparing",
        "training", "analyzing", "complete", "failed",
    ] = "pending"
    compute_backend: Optional[Literal["local", "vertex_ai"]] = None
    job_id: Optional[str] = None               # Vertex AI job name or local PID

    # Tracking
    wandb_run_id: Optional[str] = None
    wandb_run_url: Optional[str] = None

    # Results
    metrics: dict[str, Any] = {}
    artifacts: dict[str, str] = {}             # name → local path or GCS URI
    error: Optional[str] = None

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Final answer
    answer: Optional[HypothesisAnswer] = None

    @property
    def run_dir(self) -> str:
        from experimenter.config import settings
        return str(settings.runs_dir / self.id)


# ── Hypothesis answer ──────────────────────────────────────────────────────────

class HypothesisAnswer(BaseModel):
    question: str
    hypothesis: str
    verdict: Literal["confirmed", "rejected", "inconclusive"]
    confidence: float                           # 0.0 – 1.0
    evidence_summary: str                       # 2-4 sentence narrative
    key_metrics: dict[str, float]
    feature_importance: dict[str, float] = {}  # top features driving the result
    limitations: list[str] = []
    wandb_url: Optional[str] = None
    full_report: str = ""                       # markdown report


# ── Cost estimation ────────────────────────────────────────────────────────────

class CostEstimate(BaseModel):
    compute_backend: str
    machine_type: str
    estimated_hours: float
    estimated_cost_usd: float
    breakdown: dict[str, float] = {}
    recommendation: str
