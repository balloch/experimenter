"""Tests for dataset loader (download + cache)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from experimenter.data.loader import (
    DatasetDownloadResult,
    download_dataset,
    get_cached_path,
)
from experimenter.models import DatasetCandidate


@pytest.fixture()
def kaggle_candidate():
    return DatasetCandidate(
        id="user/exercise-gpa",
        name="Exercise & GPA",
        source="kaggle",
        description="",
        url="https://www.kaggle.com/datasets/user/exercise-gpa",
    )


@pytest.fixture()
def hf_candidate():
    return DatasetCandidate(
        id="mldb/student-performance",
        name="Student Performance",
        source="huggingface",
        description="",
        url="https://huggingface.co/datasets/mldb/student-performance",
    )


class TestGetCachedPath:
    def test_returns_path_for_kaggle(self, tmp_path, kaggle_candidate):
        path = get_cached_path(kaggle_candidate, cache_dir=tmp_path)
        assert "kaggle" in str(path)
        assert "exercise-gpa" in str(path)

    def test_returns_path_for_huggingface(self, tmp_path, hf_candidate):
        path = get_cached_path(hf_candidate, cache_dir=tmp_path)
        assert str(path).startswith(str(tmp_path))


class TestDownloadDataset:
    def test_kaggle_download_calls_api(self, tmp_path, kaggle_candidate):
        """Use injected API to avoid kaggle auth-on-import."""
        mock_api = MagicMock()
        expected_dir = tmp_path / "kaggle" / "user" / "exercise-gpa"

        def fake_download(dataset, path, unzip):
            expected_dir.mkdir(parents=True, exist_ok=True)
            (expected_dir / "data.csv").write_text("a,b\n1,2\n")

        mock_api.dataset_download_files.side_effect = fake_download

        result = download_dataset(kaggle_candidate, cache_dir=tmp_path, _kaggle_api=mock_api)

        assert result.success
        assert Path(result.local_path).exists()
        mock_api.dataset_download_files.assert_called_once()

    def test_kaggle_uses_cached_if_exists(self, tmp_path, kaggle_candidate):
        """Cache check happens before API call — API should not be called."""
        cached_dir = tmp_path / "kaggle" / "user" / "exercise-gpa"
        cached_dir.mkdir(parents=True)
        (cached_dir / "data.csv").write_text("a,b\n1,2\n")

        mock_api = MagicMock()
        result = download_dataset(kaggle_candidate, cache_dir=tmp_path, _kaggle_api=mock_api)

        mock_api.dataset_download_files.assert_not_called()
        assert result.success
        assert result.cached

    def test_huggingface_download(self, tmp_path, hf_candidate):
        import pandas as pd

        mock_ds = MagicMock()
        mock_ds.__contains__ = MagicMock(side_effect=lambda k: k == "train")
        mock_ds.__getitem__ = MagicMock(
            return_value=MagicMock(to_pandas=MagicMock(return_value=pd.DataFrame({"a": [1, 2]})))
        )
        mock_load = MagicMock(return_value=mock_ds)

        result = download_dataset(hf_candidate, cache_dir=tmp_path, _load_dataset_fn=mock_load)

        assert result.success
        mock_load.assert_called_once_with("mldb/student-performance")

    def test_url_copy_of_local_file(self, tmp_path, regression_csv):
        candidate = DatasetCandidate(
            id="local-file",
            name="Local",
            source="url",
            description="",
            url=str(regression_csv),
        )
        result = download_dataset(candidate, cache_dir=tmp_path)
        assert result.success
        assert Path(result.local_path).exists()

    def test_returns_failure_on_api_error(self, tmp_path, kaggle_candidate):
        mock_api = MagicMock()
        mock_api.dataset_download_files.side_effect = Exception("404 Not Found")

        result = download_dataset(kaggle_candidate, cache_dir=tmp_path, _kaggle_api=mock_api)

        assert not result.success
        assert result.error


class TestDatasetDownloadResult:
    def test_success_result(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a\n1\n")
        result = DatasetDownloadResult(success=True, local_path=str(f))
        assert result.success
        assert not result.cached

    def test_failure_result(self):
        result = DatasetDownloadResult(success=False, error="Connection refused")
        assert not result.success
        assert result.local_path is None
