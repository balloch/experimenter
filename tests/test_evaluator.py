"""Tests for hypothesis evaluator."""

import json

import pytest

from experimenter.analysis.evaluator import evaluate_results, StatisticalTest


class TestEvaluateResults:
    def test_regression_confirms_strong_correlation(self, tmp_path):
        metrics = {
            "r2": 0.78,
            "rmse": 0.25,
            "feature_importance": {"hours_exercise": 0.7, "age": 0.3},
        }
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps(metrics))

        result = evaluate_results(
            metrics_path=str(metrics_file),
            task_type="regression",
            hypothesis="Exercise frequency positively predicts GPA",
            target_variable="gpa",
            primary_feature="hours_exercise",
        )

        assert result.verdict == "confirmed"
        assert result.confidence >= 0.7

    def test_regression_rejects_weak_correlation(self, tmp_path):
        metrics = {
            "r2": 0.02,
            "rmse": 0.9,
            "feature_importance": {"hours_exercise": 0.1, "age": 0.9},
        }
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps(metrics))

        result = evaluate_results(
            metrics_path=str(metrics_file),
            task_type="regression",
            hypothesis="Exercise frequency positively predicts GPA",
            target_variable="gpa",
            primary_feature="hours_exercise",
        )

        assert result.verdict in ("rejected", "inconclusive")

    def test_classification_confirms_high_accuracy(self, tmp_path):
        metrics = {
            "accuracy": 0.88,
            "f1": 0.86,
            "feature_importance": {"income": 0.8, "age": 0.2},
        }
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps(metrics))

        result = evaluate_results(
            metrics_path=str(metrics_file),
            task_type="binary_classification",
            hypothesis="Income predicts life satisfaction",
            target_variable="satisfied",
            primary_feature="income",
        )

        assert result.verdict == "confirmed"
        assert result.confidence >= 0.7

    def test_classification_inconclusive_near_random(self, tmp_path):
        metrics = {"accuracy": 0.52, "f1": 0.50}
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps(metrics))

        result = evaluate_results(
            metrics_path=str(metrics_file),
            task_type="binary_classification",
            hypothesis="Income predicts satisfaction",
            target_variable="satisfied",
            primary_feature="income",
        )

        assert result.verdict in ("rejected", "inconclusive")

    def test_returns_key_metrics(self, tmp_path):
        metrics = {"r2": 0.65, "rmse": 0.4}
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps(metrics))

        result = evaluate_results(
            metrics_path=str(metrics_file),
            task_type="regression",
            hypothesis="h",
            target_variable="t",
            primary_feature="f",
        )

        assert "r2" in result.key_metrics

    def test_returns_feature_importance_when_available(self, tmp_path):
        metrics = {
            "r2": 0.7, "rmse": 0.3,
            "feature_importance": {"f1": 0.6, "f2": 0.4},
        }
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps(metrics))

        result = evaluate_results(
            metrics_path=str(metrics_file),
            task_type="regression",
            hypothesis="h",
            target_variable="t",
            primary_feature="f1",
        )

        assert result.feature_importance.get("f1", 0) > result.feature_importance.get("f2", 0)

    def test_includes_limitations_for_small_r2(self, tmp_path):
        metrics = {"r2": 0.15, "rmse": 0.8}
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps(metrics))

        result = evaluate_results(
            metrics_path=str(metrics_file),
            task_type="regression",
            hypothesis="h",
            target_variable="t",
            primary_feature="f",
        )

        assert len(result.limitations) > 0

    def test_confidence_is_between_0_and_1(self, tmp_path):
        for r2 in [0.0, 0.3, 0.6, 0.9]:
            metrics = {"r2": r2, "rmse": 0.5}
            metrics_file = tmp_path / f"metrics_{r2}.json"
            metrics_file.write_text(json.dumps(metrics))
            result = evaluate_results(
                metrics_path=str(metrics_file),
                task_type="regression",
                hypothesis="h",
                target_variable="t",
                primary_feature="f",
            )
            assert 0.0 <= result.confidence <= 1.0


class TestStatisticalTest:
    def test_r2_threshold_confirmed(self):
        assert StatisticalTest.regression_verdict(r2=0.6) == "confirmed"

    def test_r2_threshold_inconclusive(self):
        assert StatisticalTest.regression_verdict(r2=0.15) == "inconclusive"

    def test_r2_threshold_rejected(self):
        assert StatisticalTest.regression_verdict(r2=0.02) == "rejected"

    def test_accuracy_threshold_confirmed(self):
        assert StatisticalTest.classification_verdict(accuracy=0.8, baseline=0.5) == "confirmed"

    def test_accuracy_near_baseline_inconclusive(self):
        assert StatisticalTest.classification_verdict(accuracy=0.53, baseline=0.5) in (
            "inconclusive", "rejected"
        )
