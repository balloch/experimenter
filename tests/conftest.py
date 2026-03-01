"""Shared pytest fixtures."""

import json
import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from experimenter.models import (
    DatasetCandidate,
    ExperimentPlan,
    ExperimentRun,
)


# ── Environment ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure tests don't read real credentials from the environment."""
    for key in [
        "ANTHROPIC_API_KEY",
        "KAGGLE_USERNAME",
        "KAGGLE_KEY",
        "WANDB_API_KEY",
        "GCP_PROJECT_ID",
        "GCS_BUCKET",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ]:
        monkeypatch.delenv(key, raising=False)
    # Point data dir at a temp location so tests stay hermetic
    tmp = tempfile.mkdtemp()
    monkeypatch.setenv("EXPERIMENTER_DATA_DIR", tmp)
    monkeypatch.setenv("DEFAULT_COMPUTE", "local")


@pytest.fixture()
def tmp_data_dir(tmp_path: Path) -> Path:
    return tmp_path


# ── Sample dataframes ─────────────────────────────────────────────────────────

@pytest.fixture()
def regression_csv(tmp_path: Path) -> Path:
    """Small CSV suitable for a regression task."""
    df = pd.DataFrame(
        {
            "age": [25, 32, 45, 28, 51, 38, 60, 22, 44, 33],
            "hours_exercise": [3.0, 5.5, 2.0, 7.0, 1.5, 4.0, 0.5, 6.0, 3.5, 5.0],
            "gpa": [3.2, 3.7, 2.8, 3.9, 2.5, 3.4, 2.1, 3.8, 3.1, 3.6],
        }
    )
    path = tmp_path / "regression.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def classification_csv(tmp_path: Path) -> Path:
    """Small CSV suitable for a binary classification task."""
    df = pd.DataFrame(
        {
            "income": [50000, 80000, 30000, 120000, 45000, 95000, 25000, 75000],
            "age": [30, 45, 22, 55, 28, 48, 20, 40],
            "satisfied": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    path = tmp_path / "classification.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def csv_with_missing(tmp_path: Path) -> Path:
    """CSV with some missing values."""
    df = pd.DataFrame(
        {
            "feature_a": [1.0, None, 3.0, 4.0, None],
            "feature_b": ["x", "y", None, "x", "y"],
            "target": [10, 20, 30, 40, 50],
        }
    )
    path = tmp_path / "missing.csv"
    df.to_csv(path, index=False)
    return path


# ── Domain objects ─────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_plan() -> ExperimentPlan:
    return ExperimentPlan(
        question="Does exercise frequency predict academic performance?",
        hypothesis="Higher exercise frequency correlates with higher GPA.",
        target_variable="gpa",
        feature_variables=["hours_exercise", "age"],
        task_type="regression",
        dataset_query="student exercise GPA academic performance",
        model_type="xgboost",
        compute="local",
        time_budget_minutes=5,
    )


@pytest.fixture()
def sample_dataset_candidate() -> DatasetCandidate:
    return DatasetCandidate(
        id="user/exercise-gpa",
        name="Exercise & GPA Dataset",
        source="kaggle",
        description="Student exercise habits and GPA records.",
        url="https://www.kaggle.com/datasets/user/exercise-gpa",
        size_mb=2.5,
        num_rows=1000,
        relevance_score=0.9,
    )


@pytest.fixture()
def sample_run(sample_plan: ExperimentPlan, tmp_path: Path) -> ExperimentRun:
    run = ExperimentRun(plan=sample_plan)
    run_dir = tmp_path / run.id
    run_dir.mkdir()
    return run


@pytest.fixture()
def metrics_json(tmp_path: Path) -> Path:
    """A metrics.json file as produced by a training script."""
    metrics = {"r2": 0.72, "rmse": 0.31, "feature_importance": {"hours_exercise": 0.6, "age": 0.4}}
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics))
    return path
