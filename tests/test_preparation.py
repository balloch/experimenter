"""Tests for data preparation pipeline."""

import pandas as pd
import pytest

from experimenter.data.preparation import prepare_data, load_prepared_data


class TestPrepareData:
    def test_regression_returns_profile(self, regression_csv):
        result = prepare_data(
            dataset_path=str(regression_csv),
            target_column="gpa",
            task_type="regression",
        )
        assert result.profile.num_rows == 10
        assert result.profile.target_column == "gpa"
        assert result.prepared_path.endswith(".csv")

    def test_classification_encodes_target(self, classification_csv):
        result = prepare_data(
            dataset_path=str(classification_csv),
            target_column="satisfied",
            task_type="binary_classification",
        )
        assert result.profile.recommended_task_type in (
            "binary_classification", "regression"
        )

    def test_missing_values_are_imputed(self, csv_with_missing):
        result = prepare_data(
            dataset_path=str(csv_with_missing),
            target_column="target",
            task_type="regression",
        )
        df = pd.read_csv(result.prepared_path)
        assert df.isnull().sum().sum() == 0, "Prepared data should have no missing values"

    def test_categorical_columns_are_encoded(self, csv_with_missing):
        result = prepare_data(
            dataset_path=str(csv_with_missing),
            target_column="target",
            task_type="regression",
        )
        df = pd.read_csv(result.prepared_path)
        # All columns must be numeric after prep
        for col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"

    def test_returns_train_test_paths(self, regression_csv):
        result = prepare_data(
            dataset_path=str(regression_csv),
            target_column="gpa",
            task_type="regression",
        )
        assert result.train_path
        assert result.test_path

    def test_train_larger_than_test(self, regression_csv):
        result = prepare_data(
            dataset_path=str(regression_csv),
            target_column="gpa",
            task_type="regression",
        )
        train_df = pd.read_csv(result.train_path)
        test_df = pd.read_csv(result.test_path)
        assert len(train_df) > len(test_df)

    def test_feature_columns_filter(self, regression_csv):
        result = prepare_data(
            dataset_path=str(regression_csv),
            target_column="gpa",
            task_type="regression",
            feature_columns=["hours_exercise"],
        )
        train_df = pd.read_csv(result.train_path)
        # Only the requested feature + target
        assert "age" not in train_df.columns

    def test_target_column_missing_raises(self, regression_csv):
        with pytest.raises(ValueError, match="target"):
            prepare_data(
                dataset_path=str(regression_csv),
                target_column="nonexistent_column",
                task_type="regression",
            )

    def test_profile_reports_null_percentages(self, csv_with_missing):
        result = prepare_data(
            dataset_path=str(csv_with_missing),
            target_column="target",
            task_type="regression",
        )
        # feature_a has 2/5 = 40% nulls in raw data
        assert result.profile.columns["feature_a"].null_pct > 0

    def test_profile_warns_on_small_dataset(self, regression_csv):
        result = prepare_data(
            dataset_path=str(regression_csv),
            target_column="gpa",
            task_type="regression",
        )
        # 10 rows is very small — should warn
        assert any("small" in w.lower() or "row" in w.lower() for w in result.profile.warnings)


class TestLoadPreparedData:
    def test_load_returns_dataframes(self, regression_csv):
        result = prepare_data(str(regression_csv), "gpa", "regression")
        X_train, X_test, y_train, y_test = load_prepared_data(result)
        assert len(X_train) > 0
        assert "gpa" not in X_train.columns
        assert len(y_train) == len(X_train)
