"""Local training runner — runs training scripts in-process."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class LocalRunResult:
    success: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    model_path: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def from_metrics_file(cls, path: str) -> "LocalRunResult":
        try:
            data = json.loads(Path(path).read_text())
            return cls(success=True, metrics=data)
        except Exception as exc:
            return cls(success=False, error=str(exc))


def run_locally(
    script: str,
    data_path: str,
    target_column: str,
    task_type: str,
    output_dir: str,
    model_type: str = "xgboost",
    wandb_disabled: bool = False,
    n_estimators: int = 300,
    wandb_project: str = "experimenter",
    wandb_run_name: Optional[str] = None,
) -> LocalRunResult:
    """
    Run a training script directly in the current process.

    script: "tabular" | "nlp"
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if script == "tabular":
        return _run_tabular(
            data_path=data_path,
            target_column=target_column,
            task_type=task_type,
            output_dir=output_dir,
            model_type=model_type,
            n_estimators=n_estimators,
            wandb_disabled=wandb_disabled,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
        )
    elif script == "nlp":
        return _run_nlp(
            data_path=data_path,
            target_column=target_column,
            task_type=task_type,
            output_dir=output_dir,
            wandb_disabled=wandb_disabled,
        )
    else:
        return LocalRunResult(success=False, error=f"Unknown script: {script}")


# ── Tabular training ───────────────────────────────────────────────────────────

def _run_tabular(
    data_path: str,
    target_column: str,
    task_type: str,
    output_dir: str,
    model_type: str,
    n_estimators: int,
    wandb_disabled: bool,
    wandb_project: str,
    wandb_run_name: Optional[str],
) -> LocalRunResult:
    try:
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_squared_error,
            r2_score,
        )

        df = pd.read_csv(data_path)
        if target_column not in df.columns:
            return LocalRunResult(
                success=False, error=f"Column '{target_column}' not in dataset"
            )

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encode categoricals
        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = _build_model(model_type, task_type, n_estimators)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics: dict[str, Any] = {}
        if task_type in ("binary_classification", "multiclass_classification"):
            metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            metrics["f1"] = float(
                f1_score(y_test, y_pred, average="weighted", zero_division=0)
            )
        else:
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            metrics["r2"] = float(r2_score(y_test, y_pred))

        # Feature importance
        if hasattr(model, "feature_importances_"):
            metrics["feature_importance"] = {
                col: float(imp)
                for col, imp in zip(X.columns, model.feature_importances_)
            }

        # Save model
        out = Path(output_dir)
        model_path = str(out / "model.json")
        _save_model(model, model_path, model_type)

        metrics_path = str(out / "metrics.json")
        Path(metrics_path).write_text(json.dumps(metrics, indent=2))

        # W&B logging (optional)
        if not wandb_disabled:
            _log_wandb(metrics, model_path, wandb_project, wandb_run_name)

        logger.info("Local training complete. Metrics: %s", metrics)
        return LocalRunResult(success=True, metrics=metrics, model_path=model_path)

    except Exception as exc:
        logger.error("Local training failed: %s", exc, exc_info=True)
        return LocalRunResult(success=False, error=str(exc))


def _build_model(model_type: str, task_type: str, n_estimators: int):
    is_clf = task_type in ("binary_classification", "multiclass_classification")

    if model_type == "xgboost":
        import xgboost as xgb

        return (
            xgb.XGBClassifier(n_estimators=n_estimators, eval_metric="logloss", verbosity=0)
            if is_clf
            else xgb.XGBRegressor(n_estimators=n_estimators, verbosity=0)
        )
    elif model_type == "lightgbm":
        import lightgbm as lgb

        return (
            lgb.LGBMClassifier(n_estimators=n_estimators, verbose=-1)
            if is_clf
            else lgb.LGBMRegressor(n_estimators=n_estimators, verbose=-1)
        )
    elif model_type in ("random_forest",):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        return (
            RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
            if is_clf
            else RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1)
        )
    elif model_type in ("logistic_regression", "linear_regression"):
        if is_clf:
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(max_iter=1000)
        else:
            from sklearn.linear_model import LinearRegression

            return LinearRegression()
    else:
        # Default: XGBoost
        import xgboost as xgb

        return (
            xgb.XGBClassifier(n_estimators=n_estimators, eval_metric="logloss", verbosity=0)
            if is_clf
            else xgb.XGBRegressor(n_estimators=n_estimators, verbosity=0)
        )


def _save_model(model, path: str, model_type: str) -> None:
    if model_type in ("xgboost", "lightgbm") and hasattr(model, "save_model"):
        model.save_model(path)
    else:
        import pickle

        pkl_path = path.replace(".json", ".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)


def _log_wandb(
    metrics: dict,
    model_path: str,
    project: str,
    run_name: Optional[str],
) -> None:
    try:
        import wandb

        run = wandb.init(project=project, name=run_name, reinit=True)
        wandb.log(metrics)
        wandb.save(model_path)
        run.finish()
    except Exception as exc:
        logger.warning("W&B logging failed (non-fatal): %s", exc)


# ── NLP training (stub — uses simple TF-IDF + LR) ─────────────────────────────

def _run_nlp(
    data_path: str,
    target_column: str,
    task_type: str,
    output_dir: str,
    wandb_disabled: bool,
) -> LocalRunResult:
    try:
        import numpy as np
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(data_path)
        if target_column not in df.columns:
            return LocalRunResult(
                success=False, error=f"Column '{target_column}' not in dataset"
            )

        # Find text column (first object column that's not target)
        text_cols = [
            c for c in df.select_dtypes(include=["object"]).columns if c != target_column
        ]
        if not text_cols:
            return LocalRunResult(success=False, error="No text column found for NLP task")

        text_col = text_cols[0]
        X_text = df[text_col].fillna("").astype(str)
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y, test_size=0.2, random_state=42
        )

        vec = TfidfVectorizer(max_features=5000)
        X_tr = vec.fit_transform(X_train)
        X_te = vec.transform(X_test)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)

        metrics: dict[str, Any] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        }

        out = Path(output_dir)
        metrics_path = str(out / "metrics.json")
        Path(metrics_path).write_text(json.dumps(metrics, indent=2))

        return LocalRunResult(success=True, metrics=metrics)

    except Exception as exc:
        logger.error("NLP training failed: %s", exc, exc_info=True)
        return LocalRunResult(success=False, error=str(exc))
