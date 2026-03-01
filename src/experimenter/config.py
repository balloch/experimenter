"""Application configuration loaded from environment / .env file."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Anthropic ──────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    planner_model: str = Field(default="claude-opus-4-6", alias="PLANNER_MODEL")
    reporter_model: str = Field(default="claude-sonnet-4-6", alias="REPORTER_MODEL")

    # ── Kaggle ─────────────────────────────────────────────────────────────────
    kaggle_username: str = Field(default="", alias="KAGGLE_USERNAME")
    kaggle_key: str = Field(default="", alias="KAGGLE_KEY")

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb_api_key: str = Field(default="", alias="WANDB_API_KEY")
    wandb_project: str = Field(default="experimenter", alias="WANDB_PROJECT")

    # ── GCP ───────────────────────────────────────────────────────────────────
    gcp_project_id: str = Field(default="", alias="GCP_PROJECT_ID")
    gcp_region: str = Field(default="us-central1", alias="GCP_REGION")
    gcs_bucket: str = Field(default="", alias="GCS_BUCKET")
    google_application_credentials: str = Field(
        default="", alias="GOOGLE_APPLICATION_CREDENTIALS"
    )

    # ── Local ─────────────────────────────────────────────────────────────────
    experimenter_data_dir: Path = Field(
        default=Path.home() / ".experimenter", alias="EXPERIMENTER_DATA_DIR"
    )
    default_compute: Literal["local", "vertex_ai", "auto"] = Field(
        default="auto", alias="DEFAULT_COMPUTE"
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def runs_dir(self) -> Path:
        return self.experimenter_data_dir / "runs"

    @property
    def cache_dir(self) -> Path:
        return self.experimenter_data_dir / "cache"

    @property
    def has_gcp(self) -> bool:
        return bool(self.gcp_project_id and self.gcs_bucket)

    @property
    def has_kaggle(self) -> bool:
        return bool(self.kaggle_username and self.kaggle_key)

    @property
    def has_wandb(self) -> bool:
        return bool(self.wandb_api_key)

    def ensure_dirs(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
