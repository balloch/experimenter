"""Tests for the experiment planner (uses Claude API — mocked)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from experimenter.models import ExperimentPlan
from experimenter.planner import plan_experiment, _parse_plan_from_llm_response


MOCK_LLM_PLAN = {
    "hypothesis": "Higher exercise frequency correlates with higher GPA.",
    "target_variable": "gpa",
    "feature_variables": ["hours_exercise", "age"],
    "task_type": "regression",
    "dataset_query": "student exercise hours GPA academic performance",
    "model_type": "xgboost",
    "needs_gpu": False,
    "machine_type": "n1-standard-4",
    "description": (
        "Train an XGBoost regressor on student exercise/GPA data "
        "to test whether exercise frequency predicts academic performance."
    ),
}


class TestPlanExperiment:
    @patch("experimenter.planner.Anthropic")
    def test_returns_experiment_plan(self, MockAnthropic):
        _setup_mock_claude(MockAnthropic, MOCK_LLM_PLAN)

        plan = plan_experiment(
            question="Does exercise frequency predict academic performance?",
            time_budget_minutes=30,
        )

        assert isinstance(plan, ExperimentPlan)
        assert plan.task_type == "regression"
        assert plan.target_variable == "gpa"

    @patch("experimenter.planner.Anthropic")
    def test_passes_time_budget_to_plan(self, MockAnthropic):
        _setup_mock_claude(MockAnthropic, MOCK_LLM_PLAN)

        plan = plan_experiment(
            question="Does income predict happiness?",
            time_budget_minutes=15,
        )

        assert plan.time_budget_minutes == 15

    @patch("experimenter.planner.Anthropic")
    def test_preserves_original_question(self, MockAnthropic):
        _setup_mock_claude(MockAnthropic, MOCK_LLM_PLAN)
        question = "Does exercise frequency predict academic performance?"

        plan = plan_experiment(question=question, time_budget_minutes=30)

        assert plan.question == question

    @patch("experimenter.planner.Anthropic")
    def test_plan_has_generated_id(self, MockAnthropic):
        _setup_mock_claude(MockAnthropic, MOCK_LLM_PLAN)

        plan = plan_experiment(question="q?", time_budget_minutes=5)

        assert plan.id
        assert len(plan.id) > 0

    @patch("experimenter.planner.Anthropic")
    def test_two_plans_for_same_question_have_different_ids(self, MockAnthropic):
        _setup_mock_claude(MockAnthropic, MOCK_LLM_PLAN)

        p1 = plan_experiment(question="q?", time_budget_minutes=5)
        p2 = plan_experiment(question="q?", time_budget_minutes=5)

        assert p1.id != p2.id

    @patch("experimenter.planner.Anthropic")
    def test_calls_claude_api_once(self, MockAnthropic):
        mock_client = _setup_mock_claude(MockAnthropic, MOCK_LLM_PLAN)

        plan_experiment(question="q?", time_budget_minutes=5)

        mock_client.messages.create.assert_called_once()

    @patch("experimenter.planner.Anthropic")
    def test_includes_question_in_prompt(self, MockAnthropic):
        mock_client = _setup_mock_claude(MockAnthropic, MOCK_LLM_PLAN)
        question = "Does sleep duration affect test scores?"

        plan_experiment(question=question, time_budget_minutes=10)

        call_kwargs = mock_client.messages.create.call_args
        # The question should appear in messages or system prompt
        messages = call_kwargs.kwargs.get("messages") or (call_kwargs.args[0] if call_kwargs.args else [])
        system = call_kwargs.kwargs.get("system", "")
        prompt_text = json.dumps(messages) + system
        assert question in prompt_text

    @patch("experimenter.planner.Anthropic")
    def test_handles_llm_json_in_code_block(self, MockAnthropic):
        """Claude sometimes wraps JSON in ```json ... ``` fences."""
        plan_json = json.dumps(MOCK_LLM_PLAN)
        wrapped = f"Here is the plan:\n```json\n{plan_json}\n```"
        _setup_mock_claude_raw(MockAnthropic, wrapped)

        plan = plan_experiment(question="q?", time_budget_minutes=5)

        assert plan.task_type == "regression"

    @patch("experimenter.planner.Anthropic")
    def test_raises_on_invalid_llm_response(self, MockAnthropic):
        _setup_mock_claude_raw(MockAnthropic, "I cannot determine that.")

        with pytest.raises(ValueError, match="plan"):
            plan_experiment(question="q?", time_budget_minutes=5)


class TestParsePlanFromLlmResponse:
    def test_parses_plain_json(self):
        raw = json.dumps(MOCK_LLM_PLAN)
        plan_data = _parse_plan_from_llm_response(raw)
        assert plan_data["task_type"] == "regression"

    def test_parses_json_in_code_fence(self):
        raw = f"```json\n{json.dumps(MOCK_LLM_PLAN)}\n```"
        plan_data = _parse_plan_from_llm_response(raw)
        assert plan_data["task_type"] == "regression"

    def test_parses_json_in_unlabeled_fence(self):
        raw = f"```\n{json.dumps(MOCK_LLM_PLAN)}\n```"
        plan_data = _parse_plan_from_llm_response(raw)
        assert plan_data["target_variable"] == "gpa"

    def test_raises_value_error_on_no_json(self):
        with pytest.raises(ValueError):
            _parse_plan_from_llm_response("no json here at all")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _setup_mock_claude(MockAnthropic, plan_dict: dict):
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=json.dumps(plan_dict))]
    mock_client.messages.create.return_value = mock_message
    MockAnthropic.return_value = mock_client
    return mock_client


def _setup_mock_claude_raw(MockAnthropic, raw_text: str):
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=raw_text)]
    mock_client.messages.create.return_value = mock_message
    MockAnthropic.return_value = mock_client
    return mock_client
