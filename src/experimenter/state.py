"""Experiment run persistence — save/load JSON on disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from experimenter.models import ExperimentRun


def _run_file(run_id: str, base_dir: Path) -> Path:
    return base_dir / run_id / "run.json"


def save_run(run: ExperimentRun, base_dir: Optional[Path] = None) -> None:
    if base_dir is None:
        from experimenter.config import settings
        base_dir = settings.runs_dir
    run_dir = base_dir / run.id
    run_dir.mkdir(parents=True, exist_ok=True)
    _run_file(run.id, base_dir).write_text(run.model_dump_json(indent=2))


def load_run(run_id: str, base_dir: Optional[Path] = None) -> Optional[ExperimentRun]:
    if base_dir is None:
        from experimenter.config import settings
        base_dir = settings.runs_dir
    path = _run_file(run_id, base_dir)
    if not path.exists():
        return None
    return ExperimentRun.model_validate_json(path.read_text())


def list_runs(base_dir: Optional[Path] = None) -> list[ExperimentRun]:
    if base_dir is None:
        from experimenter.config import settings
        base_dir = settings.runs_dir
    runs = []
    if not base_dir.exists():
        return runs
    for run_dir in sorted(base_dir.iterdir()):
        run_file = run_dir / "run.json"
        if run_file.exists():
            try:
                runs.append(ExperimentRun.model_validate_json(run_file.read_text()))
            except Exception:
                pass
    return runs


def update_run_status(
    run_id: str,
    status: str,
    metrics: Optional[dict] = None,
    error: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> None:
    run = load_run(run_id, base_dir=base_dir)
    if run is None:
        return
    run.status = status  # type: ignore[assignment]
    if metrics is not None:
        run.metrics.update(metrics)
    if error is not None:
        run.error = error
    save_run(run, base_dir=base_dir)
