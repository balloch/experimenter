"""Google Cloud Storage helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def upload_file(local_path: str, gcs_uri: str) -> str:
    """Upload a local file to GCS and return the GCS URI."""
    bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    logger.info("Uploaded %s → %s", local_path, gcs_uri)
    return gcs_uri


def download_file(gcs_uri: str, local_path: str) -> str:
    """Download a GCS object to a local file."""
    bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)
    logger.info("Downloaded %s → %s", gcs_uri, local_path)
    return local_path


def dataset_gcs_uri(experiment_id: str, filename: str = "data.csv") -> str:
    """Return a GCS URI for storing a dataset for a given experiment."""
    from experimenter.config import settings

    return f"gs://{settings.gcs_bucket}/experiments/{experiment_id}/data/{filename}"


def output_gcs_uri(experiment_id: str) -> str:
    """Return the GCS output directory URI for a Vertex AI job."""
    from experimenter.config import settings

    return f"gs://{settings.gcs_bucket}/experiments/{experiment_id}/output"


def _parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """Parse 'gs://bucket/path/to/blob' → ('bucket', 'path/to/blob')."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    parts = gcs_uri[5:].split("/", 1)
    bucket = parts[0]
    blob = parts[1] if len(parts) > 1 else ""
    return bucket, blob
