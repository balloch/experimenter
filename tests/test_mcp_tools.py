"""Tests for MCP tool definitions and server registration."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from experimenter.models import ExperimentPlan, DatasetCandidate


class TestMcpServerRegistration:
    def test_mcp_server_can_be_imported(self):
        from experimenter import mcp_server  # noqa: F401

    def test_mcp_app_has_required_tools(self):
        from experimenter.mcp_server import mcp
        # list_tools() is async in MCP 1.x
        tools = asyncio.run(mcp.list_tools())
        tool_names = {t.name for t in tools}
        required = {
            "plan_experiment",
            "search_datasets",
            "download_dataset",
            "prepare_data",
            "estimate_cost",
            "run_experiment",
            "get_experiment_status",
            "get_experiment_results",
            "answer_hypothesis",
        }
        missing = required - tool_names
        assert not missing, f"Missing MCP tools: {missing}"

    def test_each_tool_has_description(self):
        from experimenter.mcp_server import mcp
        tools = asyncio.run(mcp.list_tools())
        for tool in tools:
            assert tool.description, f"Tool '{tool.name}' has no description"

    def test_each_tool_has_input_schema(self):
        from experimenter.mcp_server import mcp
        tools = asyncio.run(mcp.list_tools())
        for tool in tools:
            assert tool.inputSchema, f"Tool '{tool.name}' has no input schema"


class TestPlanExperimentTool:
    # patch the function where it is defined (definitions.py imports it there)
    @patch("experimenter.tools.definitions.plan_experiment")
    async def test_returns_json_string(self, mock_planner, sample_plan):
        mock_planner.return_value = sample_plan
        from experimenter.tools.definitions import tool_plan_experiment

        result = await tool_plan_experiment(
            question="Does exercise predict GPA?",
            time_budget_minutes=30,
        )
        parsed = json.loads(result)
        assert "id" in parsed
        assert "question" in parsed

    @patch("experimenter.tools.definitions.plan_experiment")
    async def test_includes_human_readable_description(self, mock_planner, sample_plan):
        mock_planner.return_value = sample_plan
        from experimenter.tools.definitions import tool_plan_experiment

        result = await tool_plan_experiment(
            question="Does exercise predict GPA?",
            time_budget_minutes=30,
        )
        parsed = json.loads(result)
        assert parsed.get("description") or parsed.get("hypothesis")


class TestSearchDatasetsTool:
    @patch("experimenter.tools.definitions.search_all", new_callable=AsyncMock)
    async def test_returns_json_list(self, mock_search, sample_dataset_candidate):
        mock_search.return_value = [sample_dataset_candidate]
        from experimenter.tools.definitions import tool_search_datasets

        result = await tool_search_datasets(query="exercise gpa")
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["source"] == "kaggle"

    @patch("experimenter.tools.definitions.search_all", new_callable=AsyncMock)
    async def test_passes_sources_filter(self, mock_search):
        mock_search.return_value = []
        from experimenter.tools.definitions import tool_search_datasets

        await tool_search_datasets(query="q", sources=["kaggle"])
        mock_search.assert_called_once_with("q", sources=["kaggle"], limit_per_source=5)


class TestEstimateCostTool:
    def test_returns_json_cost_estimate(self, sample_plan):
        from experimenter.tools.definitions import tool_estimate_cost

        plan_json = sample_plan.model_dump_json()
        result = tool_estimate_cost(plan_json=plan_json, compute="local")
        parsed = json.loads(result)
        assert "estimated_cost_usd" in parsed
        assert parsed["estimated_cost_usd"] == 0.0  # local is free

    def test_vertex_ai_cost_is_positive(self, sample_plan):
        plan = sample_plan.model_copy(update={"compute": "vertex_ai", "time_budget_minutes": 30})
        from experimenter.tools.definitions import tool_estimate_cost

        result = tool_estimate_cost(plan_json=plan.model_dump_json(), compute="vertex_ai")
        parsed = json.loads(result)
        assert parsed["estimated_cost_usd"] > 0


class TestGetExperimentStatusTool:
    def test_returns_not_found_for_unknown_id(self, tmp_path):
        from experimenter.state import load_run
        from experimenter.tools.definitions import tool_get_experiment_status

        with patch("experimenter.tools.definitions.load_run", return_value=None):
            result = tool_get_experiment_status(experiment_id="nonexistent-id")
        parsed = json.loads(result)
        assert parsed["status"] == "not_found"

    def test_returns_status_for_known_run(self, sample_run, tmp_path):
        from experimenter.tools.definitions import tool_get_experiment_status

        with patch("experimenter.tools.definitions.load_run", return_value=sample_run):
            result = tool_get_experiment_status(experiment_id=sample_run.id)
        parsed = json.loads(result)
        assert parsed["status"] == "pending"
        assert parsed["id"] == sample_run.id
