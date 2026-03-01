"""Tests for local training runner."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from experimenter.cloud.local_runner import run_locally, LocalRunResult


class TestRunLocally:
    def test_runs_tabular_training_script(self, regression_csv, tmp_path):
        """Full integration test: actually trains a tiny model locally."""
        result = run_locally(
            script="tabular",
            data_path=str(regression_csv),
            target_column="gpa",
            task_type="regression",
            model_type="xgboost",
            output_dir=str(tmp_path / "output"),
            wandb_disabled=True,
            n_estimators=10,
        )
        assert result.success
        assert result.metrics.get("r2") is not None
        assert result.metrics.get("rmse") is not None

    def test_creates_metrics_json(self, regression_csv, tmp_path):
        output_dir = tmp_path / "output"
        run_locally(
            script="tabular",
            data_path=str(regression_csv),
            target_column="gpa",
            task_type="regression",
            output_dir=str(output_dir),
            wandb_disabled=True,
            n_estimators=10,
        )
        metrics_path = output_dir / "metrics.json"
        assert metrics_path.exists()

    def test_creates_model_artifact(self, regression_csv, tmp_path):
        output_dir = tmp_path / "output"
        run_locally(
            script="tabular",
            data_path=str(regression_csv),
            target_column="gpa",
            task_type="regression",
            output_dir=str(output_dir),
            wandb_disabled=True,
            n_estimators=10,
        )
        # Model file should exist
        model_files = list(output_dir.glob("model.*"))
        assert len(model_files) >= 1

    def test_returns_failure_on_bad_target(self, regression_csv, tmp_path):
        result = run_locally(
            script="tabular",
            data_path=str(regression_csv),
            target_column="nonexistent",
            task_type="regression",
            output_dir=str(tmp_path / "output"),
            wandb_disabled=True,
        )
        assert not result.success
        assert result.error

    def test_classification_produces_accuracy(self, classification_csv, tmp_path):
        result = run_locally(
            script="tabular",
            data_path=str(classification_csv),
            target_column="satisfied",
            task_type="binary_classification",
            model_type="xgboost",
            output_dir=str(tmp_path / "output"),
            wandb_disabled=True,
            n_estimators=10,
        )
        assert result.success
        assert "accuracy" in result.metrics or "f1" in result.metrics

    def test_lightgbm_model_type(self, regression_csv, tmp_path):
        result = run_locally(
            script="tabular",
            data_path=str(regression_csv),
            target_column="gpa",
            task_type="regression",
            model_type="lightgbm",
            output_dir=str(tmp_path / "output"),
            wandb_disabled=True,
            n_estimators=10,
        )
        assert result.success

    def test_result_includes_feature_importance(self, regression_csv, tmp_path):
        result = run_locally(
            script="tabular",
            data_path=str(regression_csv),
            target_column="gpa",
            task_type="regression",
            output_dir=str(tmp_path / "output"),
            wandb_disabled=True,
            n_estimators=10,
        )
        assert result.success
        assert "feature_importance" in result.metrics


class TestLocalRunResult:
    def test_from_metrics_json(self, metrics_json):
        result = LocalRunResult.from_metrics_file(str(metrics_json))
        assert result.success
        assert result.metrics["r2"] == pytest.approx(0.72)

    def test_from_missing_file_is_failure(self, tmp_path):
        result = LocalRunResult.from_metrics_file(str(tmp_path / "missing.json"))
        assert not result.success
        assert result.error
