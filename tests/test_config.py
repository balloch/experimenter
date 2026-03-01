"""Tests for application configuration."""

import os
from pathlib import Path


def test_default_compute_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DEFAULT_COMPUTE", "local")
    monkeypatch.setenv("EXPERIMENTER_DATA_DIR", str(tmp_path))
    # Re-import to pick up env changes
    from importlib import reload
    import experimenter.config as cfg_module
    reload(cfg_module)
    from experimenter.config import Settings
    s = Settings()
    assert s.default_compute == "local"


def test_has_gcp_false_when_not_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("EXPERIMENTER_DATA_DIR", str(tmp_path))
    from experimenter.config import Settings
    s = Settings()
    assert s.has_gcp is False


def test_has_gcp_true_when_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("GCP_PROJECT_ID", "my-project")
    monkeypatch.setenv("GCS_BUCKET", "my-bucket")
    monkeypatch.setenv("EXPERIMENTER_DATA_DIR", str(tmp_path))
    from experimenter.config import Settings
    s = Settings()
    assert s.has_gcp is True


def test_has_kaggle_false_when_not_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("EXPERIMENTER_DATA_DIR", str(tmp_path))
    from experimenter.config import Settings
    s = Settings()
    assert s.has_kaggle is False


def test_has_kaggle_true_when_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("KAGGLE_KEY", "key123")
    monkeypatch.setenv("EXPERIMENTER_DATA_DIR", str(tmp_path))
    from experimenter.config import Settings
    s = Settings()
    assert s.has_kaggle is True


def test_ensure_dirs_creates_directories(monkeypatch, tmp_path):
    monkeypatch.setenv("EXPERIMENTER_DATA_DIR", str(tmp_path / "exp"))
    from experimenter.config import Settings
    s = Settings()
    s.ensure_dirs()
    assert s.runs_dir.exists()
    assert s.cache_dir.exists()


def test_runs_dir_is_under_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("EXPERIMENTER_DATA_DIR", str(tmp_path))
    from experimenter.config import Settings
    s = Settings()
    assert str(s.runs_dir).startswith(str(tmp_path))
    assert "runs" in str(s.runs_dir)
