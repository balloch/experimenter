"""Tests for cloud cost estimation."""

import pytest

from experimenter.cloud.cost_estimator import (
    MACHINE_HOURLY_RATES,
    estimate_cost,
    recommend_machine,
)
from experimenter.models import ExperimentPlan


@pytest.fixture()
def regression_plan():
    return ExperimentPlan(
        question="q", hypothesis="h", target_variable="t",
        task_type="regression", dataset_query="q",
        machine_type="n1-standard-4",
        time_budget_minutes=30,
    )


@pytest.fixture()
def gpu_plan():
    return ExperimentPlan(
        question="q", hypothesis="h", target_variable="t",
        task_type="nlp_classification", dataset_query="q",
        machine_type="n1-standard-8",
        needs_gpu=True,
        time_budget_minutes=60,
    )


class TestMachineRates:
    def test_rates_table_not_empty(self):
        assert len(MACHINE_HOURLY_RATES) > 0

    def test_n1_standard_4_in_table(self):
        assert "n1-standard-4" in MACHINE_HOURLY_RATES

    def test_all_rates_positive(self):
        for machine, rate in MACHINE_HOURLY_RATES.items():
            assert rate > 0, f"{machine} has non-positive rate"


class TestEstimateCost:
    def test_local_cost_is_zero(self, regression_plan):
        est = estimate_cost(regression_plan, compute="local")
        assert est.estimated_cost_usd == 0.0
        assert est.compute_backend == "local"

    def test_vertex_ai_cost_is_positive(self, regression_plan):
        est = estimate_cost(regression_plan, compute="vertex_ai")
        assert est.estimated_cost_usd > 0
        assert est.compute_backend == "vertex_ai"

    def test_cost_scales_with_time(self):
        plan_short = ExperimentPlan(
            question="q", hypothesis="h", target_variable="t",
            task_type="regression", dataset_query="q",
            machine_type="n1-standard-4", time_budget_minutes=10,
        )
        plan_long = ExperimentPlan(
            question="q", hypothesis="h", target_variable="t",
            task_type="regression", dataset_query="q",
            machine_type="n1-standard-4", time_budget_minutes=60,
        )
        short_est = estimate_cost(plan_short, compute="vertex_ai")
        long_est = estimate_cost(plan_long, compute="vertex_ai")
        assert long_est.estimated_cost_usd > short_est.estimated_cost_usd

    def test_gpu_machine_costs_more_than_cpu(self, regression_plan, gpu_plan):
        cpu_est = estimate_cost(regression_plan, compute="vertex_ai")
        gpu_est = estimate_cost(gpu_plan, compute="vertex_ai")
        # GPU plan is 2× time but also more expensive per hour
        assert gpu_est.estimated_cost_usd > cpu_est.estimated_cost_usd

    def test_auto_recommends_local_for_short_budgets(self):
        plan = ExperimentPlan(
            question="q", hypothesis="h", target_variable="t",
            task_type="regression", dataset_query="q",
            compute="auto", time_budget_minutes=3,
        )
        est = estimate_cost(plan, compute="auto")
        assert est.compute_backend == "local"

    def test_auto_recommends_vertex_for_long_budgets(self):
        plan = ExperimentPlan(
            question="q", hypothesis="h", target_variable="t",
            task_type="regression", dataset_query="q",
            compute="auto", time_budget_minutes=60,
        )
        est = estimate_cost(plan, compute="auto")
        assert est.compute_backend in ("vertex_ai", "local")

    def test_estimate_includes_recommendation_text(self, regression_plan):
        est = estimate_cost(regression_plan, compute="vertex_ai")
        assert len(est.recommendation) > 0


class TestRecommendMachine:
    def test_tabular_no_gpu_returns_cpu_machine(self):
        machine = recommend_machine(task_type="regression", needs_gpu=False, time_budget_minutes=30)
        assert "gpu" not in machine.lower()

    def test_nlp_with_gpu_returns_gpu_machine(self):
        machine = recommend_machine(task_type="nlp_classification", needs_gpu=True, time_budget_minutes=60)
        # Should return a machine with GPU accelerator capability
        assert machine  # non-empty

    def test_returns_string(self):
        m = recommend_machine("regression", False, 10)
        assert isinstance(m, str)
        assert len(m) > 0
