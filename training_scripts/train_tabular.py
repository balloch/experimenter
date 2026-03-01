#!/usr/bin/env python
"""
Tabular ML training script.

Runs locally or inside a Vertex AI custom training container.
All arguments can be passed via CLI or environment variables.

Usage:
    python train_tabular.py \
        --data-path data.csv \
        --target-column gpa \
        --task-type regression \
        --output-dir /tmp/output
"""

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Tabular ML training script")
    p.add_argument("--data-path", required=True, help="Path to training CSV (local or GCS)")
    p.add_argument("--target-column", required=True, help="Name of target column")
    p.add_argument("--task-type", default="regression",
                   choices=["regression", "binary_classification", "multiclass_classification"])
    p.add_argument("--model-type", default="xgboost",
                   choices=["xgboost", "lightgbm", "random_forest",
                            "logistic_regression", "linear_regression"])
    p.add_argument("--output-dir", default=os.environ.get("AIP_MODEL_DIR", "/tmp/output"))
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "experimenter"))
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--no-wandb", action="store_true", default=False)
    return p.parse_args()


def load_data_from_gcs(gcs_path: str, local_path: str) -> str:
    """Download data from GCS to a local temp file if needed."""
    if not gcs_path.startswith("gs://"):
        return gcs_path
    from google.cloud import storage

    bucket_name, blob_name = gcs_path[5:].split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)
    return local_path


def main():
    args = parse_args()

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (
        accuracy_score, f1_score, mean_squared_error, r2_score,
    )

    # Resolve data path (may be GCS)
    data_path = args.data_path
    if data_path.startswith("gs://"):
        data_path = load_data_from_gcs(data_path, "/tmp/train_data.csv")

    df = pd.read_csv(data_path)
    if args.target_column not in df.columns:
        print(f"ERROR: Target column '{args.target_column}' not found.", file=sys.stderr)
        sys.exit(1)

    X = df.drop(columns=[args.target_column])
    y = df[args.target_column]

    # Encode categoricals
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    # Build model
    is_clf = args.task_type in ("binary_classification", "multiclass_classification")

    if args.model_type == "xgboost":
        import xgboost as xgb
        model = (
            xgb.XGBClassifier(n_estimators=args.n_estimators, eval_metric="logloss", verbosity=0)
            if is_clf
            else xgb.XGBRegressor(n_estimators=args.n_estimators, verbosity=0)
        )
    elif args.model_type == "lightgbm":
        import lightgbm as lgb
        model = (
            lgb.LGBMClassifier(n_estimators=args.n_estimators, verbose=-1)
            if is_clf
            else lgb.LGBMRegressor(n_estimators=args.n_estimators, verbose=-1)
        )
    elif args.model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        model = (
            RandomForestClassifier(n_estimators=args.n_estimators, n_jobs=-1)
            if is_clf
            else RandomForestRegressor(n_estimators=args.n_estimators, n_jobs=-1)
        )
    elif args.model_type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)
    else:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()

    print(f"Training {args.model_type} for {args.task_type} on {len(X_train)} samples...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    metrics = {}
    if is_clf:
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["f1"] = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        metrics["n_classes"] = int(y.nunique())
    else:
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics["r2"] = float(r2_score(y_test, y_pred))

    if hasattr(model, "feature_importances_"):
        metrics["feature_importance"] = {
            col: float(imp)
            for col, imp in zip(X.columns, model.feature_importances_)
        }

    print(f"Metrics: {metrics}")

    # Save outputs
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = str(out / "model.json")
    if hasattr(model, "save_model"):
        model.save_model(model_path)
    else:
        import pickle
        with open(str(out / "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        model_path = str(out / "model.pkl")

    metrics_path = str(out / "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # W&B logging
    if not args.no_wandb and os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
                reinit=True,
            )
            wandb.log(metrics)
            wandb.save(model_path)
            wandb.save(metrics_path)
            print(f"W&B run: {run.get_url()}")
            run.finish()
        except Exception as e:
            print(f"W&B logging failed (non-fatal): {e}", file=sys.stderr)

    # Vertex AI: copy model to AIP_MODEL_DIR if set
    aip_dir = os.environ.get("AIP_MODEL_DIR", "")
    if aip_dir and aip_dir != args.output_dir:
        import shutil
        aip_path = Path(aip_dir)
        aip_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(metrics_path, aip_path / "metrics.json")
        if Path(model_path).exists():
            shutil.copy(model_path, aip_path / Path(model_path).name)

    print("Training complete.")


if __name__ == "__main__":
    main()
