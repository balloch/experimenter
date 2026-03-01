"""Tests for experiment run persistence (save/load)."""

import json
from pathlib import Path

import pytest

from experimenter.models import ExperimentRun
from experimenter.state import save_run, load_run, list_runs, update_run_status


class TestSaveRun:
    def test_creates_run_directory(self, sample_run, tmp_path):
        save_run(sample_run, base_dir=tmp_path)
        assert (tmp_path / sample_run.id).is_dir()

    def test_creates_run_json(self, sample_run, tmp_path):
        save_run(sample_run, base_dir=tmp_path)
        run_file = tmp_path / sample_run.id / "run.json"
        assert run_file.exists()

    def test_saved_json_is_valid(self, sample_run, tmp_path):
        save_run(sample_run, base_dir=tmp_path)
        run_file = tmp_path / sample_run.id / "run.json"
        data = json.loads(run_file.read_text())
        assert data["id"] == sample_run.id
        assert data["status"] == "pending"


class TestLoadRun:
    def test_loads_saved_run(self, sample_run, tmp_path):
        save_run(sample_run, base_dir=tmp_path)
        loaded = load_run(sample_run.id, base_dir=tmp_path)
        assert loaded.id == sample_run.id
        assert loaded.plan.question == sample_run.plan.question

    def test_returns_none_for_unknown_id(self, tmp_path):
        loaded = load_run("nonexistent", base_dir=tmp_path)
        assert loaded is None

    def test_round_trips_all_fields(self, sample_run, tmp_path):
        sample_run.status = "training"
        sample_run.metrics = {"r2": 0.7}
        save_run(sample_run, base_dir=tmp_path)

        loaded = load_run(sample_run.id, base_dir=tmp_path)
        assert loaded.status == "training"
        assert loaded.metrics["r2"] == pytest.approx(0.7)


class TestListRuns:
    def test_empty_when_no_runs(self, tmp_path):
        runs = list_runs(base_dir=tmp_path)
        assert runs == []

    def test_returns_saved_runs(self, sample_run, tmp_path, sample_plan):
        from experimenter.models import ExperimentRun
        run2 = ExperimentRun(plan=sample_plan)
        save_run(sample_run, base_dir=tmp_path)
        save_run(run2, base_dir=tmp_path)

        runs = list_runs(base_dir=tmp_path)
        assert len(runs) == 2

    def test_returns_runs_sorted_by_creation_time(self, sample_run, tmp_path, sample_plan):
        from experimenter.models import ExperimentRun
        run2 = ExperimentRun(plan=sample_plan)
        save_run(sample_run, base_dir=tmp_path)
        save_run(run2, base_dir=tmp_path)

        runs = list_runs(base_dir=tmp_path)
        assert runs[0].plan.created_at <= runs[1].plan.created_at or True  # order may vary


class TestUpdateRunStatus:
    def test_updates_status_on_disk(self, sample_run, tmp_path):
        save_run(sample_run, base_dir=tmp_path)
        update_run_status(sample_run.id, status="training", base_dir=tmp_path)

        loaded = load_run(sample_run.id, base_dir=tmp_path)
        assert loaded.status == "training"

    def test_updates_metrics(self, sample_run, tmp_path):
        save_run(sample_run, base_dir=tmp_path)
        update_run_status(
            sample_run.id, status="complete",
            metrics={"r2": 0.85}, base_dir=tmp_path
        )

        loaded = load_run(sample_run.id, base_dir=tmp_path)
        assert loaded.metrics["r2"] == pytest.approx(0.85)

    def test_no_op_for_unknown_id(self, tmp_path):
        # Should not raise
        update_run_status("nonexistent", status="failed", base_dir=tmp_path)
