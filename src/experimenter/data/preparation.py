"""Data preparation: EDA, imputation, encoding, train/test split."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from experimenter.models import ColumnProfile, DataProfile

logger = logging.getLogger(__name__)

SMALL_DATASET_THRESHOLD = 100


@dataclass
class PrepResult:
    prepared_path: str
    train_path: str
    test_path: str
    profile: DataProfile


def prepare_data(
    dataset_path: str,
    target_column: str,
    task_type: str,
    feature_columns: Optional[list[str]] = None,
    test_size: float = 0.2,
    output_dir: Optional[str] = None,
) -> PrepResult:
    """
    Load CSV, impute missing values, encode categoricals, split into
    train/test CSVs, and return a PrepResult with paths and data profile.
    """
    df = pd.read_csv(dataset_path)

    if target_column not in df.columns:
        raise ValueError(
            f"target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Build raw profile before any transformation
    profile = _build_profile(df, target_column, task_type)

    # Select features
    if feature_columns:
        missing_feats = [c for c in feature_columns if c not in df.columns]
        if missing_feats:
            raise ValueError(f"Feature columns not found: {missing_feats}")
        df = df[feature_columns + [target_column]]

    # Impute + encode
    df = _impute(df)
    df = _encode_categoricals(df, target_column, task_type)

    # Determine output dir (same dir as source by default)
    if output_dir is None:
        output_dir = str(Path(dataset_path).parent / "prepared")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    prepared_path = str(Path(output_dir) / "prepared.csv")
    df.to_csv(prepared_path, index=False)

    # Train / test split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_path = str(Path(output_dir) / "train.csv")
    test_path = str(Path(output_dir) / "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return PrepResult(
        prepared_path=prepared_path,
        train_path=train_path,
        test_path=test_path,
        profile=profile,
    )


def load_prepared_data(
    result: PrepResult,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return (X_train, X_test, y_train, y_test) from a PrepResult."""
    train_df = pd.read_csv(result.train_path)
    test_df = pd.read_csv(result.test_path)
    target = result.profile.target_column

    X_train = train_df.drop(columns=[target])
    X_test = test_df.drop(columns=[target])
    y_train = train_df[target]
    y_test = test_df[target]
    return X_train, X_test, y_train, y_test


# ── Private helpers ────────────────────────────────────────────────────────────

def _build_profile(df: pd.DataFrame, target_column: str, task_type: str) -> DataProfile:
    columns: dict[str, ColumnProfile] = {}
    for col in df.columns:
        null_pct = float(df[col].isna().mean())
        sample = [v for v in df[col].dropna().head(5).tolist() if v is not None]
        columns[col] = ColumnProfile(
            dtype=str(df[col].dtype),
            null_pct=null_pct,
            unique_count=int(df[col].nunique()),
            sample_values=sample,
        )

    target = df[target_column]
    if task_type in ("regression", "time_series"):
        target_dist: dict = {
            "mean": float(target.mean()),
            "std": float(target.std()),
            "min": float(target.min()),
            "max": float(target.max()),
        }
    else:
        target_dist = {str(k): int(v) for k, v in target.value_counts().items()}

    warnings: list[str] = []
    if len(df) < SMALL_DATASET_THRESHOLD:
        warnings.append(
            f"Small dataset: only {len(df)} rows. Results may not be reliable."
        )

    return DataProfile(
        num_rows=len(df),
        num_columns=len(df.columns),
        columns=columns,
        target_column=target_column,
        target_distribution=target_dist,
        recommended_task_type=task_type,
        warnings=warnings,
    )


def _impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "unknown")
    return df


def _encode_categoricals(df: pd.DataFrame, target_column: str, task_type: str) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df
