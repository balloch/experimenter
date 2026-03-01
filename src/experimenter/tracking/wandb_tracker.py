"""Weights & Biases experiment tracking."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def init_run(
    project: str,
    run_name: Optional[str] = None,
    config: Optional[dict] = None,
    tags: Optional[list[str]] = None,
) -> Optional[Any]:
    """
    Initialize a W&B run. Returns the run object or None if W&B is unavailable.
    """
    try:
        import wandb

        run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            tags=tags or [],
            reinit=True,
        )
        return run
    except Exception as exc:
        logger.warning("W&B init failed (non-fatal): %s", exc)
        return None


def log_metrics(metrics: dict[str, Any], step: Optional[int] = None) -> None:
    try:
        import wandb

        wandb.log(metrics, step=step)
    except Exception as exc:
        logger.warning("W&B log_metrics failed: %s", exc)


def finish_run() -> Optional[str]:
    """Finish the current W&B run and return its URL."""
    try:
        import wandb

        run = wandb.run
        if run is None:
            return None
        url = run.get_url()
        run.finish()
        return url
    except Exception as exc:
        logger.warning("W&B finish failed: %s", exc)
        return None


def get_run_url() -> Optional[str]:
    try:
        import wandb

        if wandb.run:
            return wandb.run.get_url()
        return None
    except Exception:
        return None
