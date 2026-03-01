"""Tests for Pydantic data models."""

import pytest
from pydantic import ValidationError

from experimenter.models import (
    CostEstimate,
    DataProfile,
    DatasetCandidate,
    ExperimentPlan,
    ExperimentRun,
    HypothesisAnswer,
    ColumnProfile,
)


class TestDatasetCandidate:
    def test_valid_kaggle_candidate(self):
        c = DatasetCandidate(
            id="user/ds",
            name="My Dataset",
            source="kaggle",
            description="desc",
            url="https://kaggle.com/datasets/user/ds",
        )
        assert c.source == "kaggle"
        assert c.relevance_score == 0.0

    def test_invalid_source_raises(self):
        with pytest.raises(ValidationError):
            DatasetCandidate(
                id="x", name="x", source="s3", description="", url="http://x"
            )

    def test_valid_sources(self):
        for src in ("kaggle", "huggingface", "url", "web"):
            c = DatasetCandidate(id="x", name="x", source=src, description="", url="http://x")
            assert c.source == src


class TestExperimentPlan:
    def test_creates_with_defaults(self, sample_plan):
        assert sample_plan.id  # auto-generated UUID
        assert sample_plan.model_type == "xgboost"
        assert sample_plan.compute == "local"

    def test_invalid_task_type_raises(self):
        with pytest.raises(ValidationError):
            ExperimentPlan(
                question="q",
                hypothesis="h",
                target_variable="t",
                task_type="banana_classification",
                dataset_query="q",
            )

    def test_valid_task_types(self):
        valid = [
            "binary_classification",
            "multiclass_classification",
            "regression",
            "nlp_classification",
            "nlp_regression",
            "clustering",
            "time_series",
        ]
        for tt in valid:
            p = ExperimentPlan(
                question="q", hypothesis="h", target_variable="t",
                task_type=tt, dataset_query="q",
            )
            assert p.task_type == tt

    def test_two_plans_have_different_ids(self):
        p1 = ExperimentPlan(question="q", hypothesis="h", target_variable="t",
                            task_type="regression", dataset_query="q")
        p2 = ExperimentPlan(question="q", hypothesis="h", target_variable="t",
                            task_type="regression", dataset_query="q")
        assert p1.id != p2.id


class TestExperimentRun:
    def test_initial_status_is_pending(self, sample_plan):
        run = ExperimentRun(plan=sample_plan)
        assert run.status == "pending"

    def test_run_has_unique_id(self, sample_plan):
        r1 = ExperimentRun(plan=sample_plan)
        r2 = ExperimentRun(plan=sample_plan)
        assert r1.id != r2.id

    def test_invalid_status_raises(self, sample_plan):
        with pytest.raises(ValidationError):
            ExperimentRun(plan=sample_plan, status="running")

    def test_metrics_default_empty(self, sample_plan):
        run = ExperimentRun(plan=sample_plan)
        assert run.metrics == {}
        assert run.artifacts == {}


class TestHypothesisAnswer:
    def test_valid_answer(self):
        a = HypothesisAnswer(
            question="q",
            hypothesis="h",
            verdict="confirmed",
            confidence=0.85,
            evidence_summary="Strong positive correlation found.",
            key_metrics={"r2": 0.72},
        )
        assert a.verdict == "confirmed"
        assert 0 <= a.confidence <= 1

    def test_invalid_verdict_raises(self):
        with pytest.raises(ValidationError):
            HypothesisAnswer(
                question="q", hypothesis="h", verdict="maybe",
                confidence=0.5, evidence_summary="x", key_metrics={},
            )

    def test_valid_verdicts(self):
        for verdict in ("confirmed", "rejected", "inconclusive"):
            a = HypothesisAnswer(
                question="q", hypothesis="h", verdict=verdict,
                confidence=0.5, evidence_summary="x", key_metrics={},
            )
            assert a.verdict == verdict


class TestCostEstimate:
    def test_creates_cost_estimate(self):
        est = CostEstimate(
            compute_backend="vertex_ai",
            machine_type="n1-standard-4",
            estimated_hours=0.5,
            estimated_cost_usd=0.095,
            recommendation="Use spot instances for 60% savings.",
        )
        assert est.estimated_cost_usd == 0.095
        assert est.breakdown == {}
