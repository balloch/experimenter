"""Dataset downloader with local cache."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from experimenter.models import DatasetCandidate

logger = logging.getLogger(__name__)

# datasets is safe to import at module level (no auto-auth)
from datasets import load_dataset


@dataclass
class DatasetDownloadResult:
    success: bool
    local_path: Optional[str] = None
    cached: bool = False
    error: Optional[str] = None


def get_cached_path(candidate: DatasetCandidate, cache_dir: Path) -> Path:
    """Return the expected local directory for a given dataset candidate."""
    if candidate.source == "kaggle":
        # id is like "owner/dataset-slug"
        safe_id = candidate.id.replace("/", "_")
        owner, *slug_parts = candidate.id.split("/")
        slug = "_".join(slug_parts) if slug_parts else safe_id
        return cache_dir / "kaggle" / owner / slug
    elif candidate.source == "huggingface":
        safe_id = candidate.id.replace("/", "_")
        return cache_dir / "huggingface" / safe_id
    else:
        import hashlib
        url_hash = hashlib.md5(candidate.url.encode()).hexdigest()[:12]
        return cache_dir / "url" / url_hash


def _find_csv(directory: Path) -> Optional[Path]:
    """Return the first CSV found in directory (recursive)."""
    csvs = sorted(directory.rglob("*.csv"))
    return csvs[0] if csvs else None


def download_dataset(
    candidate: DatasetCandidate,
    cache_dir: Optional[Path] = None,
    _kaggle_api=None,
    _load_dataset_fn=None,
) -> DatasetDownloadResult:
    if cache_dir is None:
        from experimenter.config import settings
        cache_dir = settings.cache_dir

    cached_dir = get_cached_path(candidate, cache_dir)

    # Return cached result if already downloaded
    if cached_dir.exists():
        csv = _find_csv(cached_dir)
        if csv:
            logger.info("Using cached dataset at %s", csv)
            return DatasetDownloadResult(success=True, local_path=str(csv), cached=True)

    if candidate.source == "kaggle":
        return _download_kaggle(candidate, cached_dir, _api=_kaggle_api)
    elif candidate.source == "huggingface":
        return _download_huggingface(candidate, cached_dir, _load_fn=_load_dataset_fn)
    elif candidate.source in ("url", "web"):
        return _download_url(candidate, cached_dir)
    else:
        return DatasetDownloadResult(success=False, error=f"Unknown source: {candidate.source}")


def _download_kaggle(
    candidate: DatasetCandidate, dest_dir: Path, _api=None
) -> DatasetDownloadResult:
    try:
        if _api is None:
            import kaggle.api.kaggle_api_extended as _m  # noqa: PLC0415

            _api = _m.KaggleApiExtended()
            _api.authenticate()
        dest_dir.mkdir(parents=True, exist_ok=True)
        _api.dataset_download_files(candidate.id, path=str(dest_dir), unzip=True)
        csv = _find_csv(dest_dir)
        if csv is None:
            return DatasetDownloadResult(
                success=False, error="Download succeeded but no CSV found"
            )
        return DatasetDownloadResult(success=True, local_path=str(csv))
    except Exception as exc:
        logger.error("Kaggle download failed: %s", exc)
        return DatasetDownloadResult(success=False, error=str(exc))


def _download_huggingface(
    candidate: DatasetCandidate, dest_dir: Path, _load_fn=None
) -> DatasetDownloadResult:
    try:
        if _load_fn is None:
            _load_fn = load_dataset
        ds = _load_fn(candidate.id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        split_name = "train" if "train" in ds else list(ds.keys())[0]
        df = ds[split_name].to_pandas()
        csv_path = dest_dir / "data.csv"
        df.to_csv(csv_path, index=False)
        return DatasetDownloadResult(success=True, local_path=str(csv_path))
    except Exception as exc:
        logger.error("HuggingFace download failed: %s", exc)
        return DatasetDownloadResult(success=False, error=str(exc))


def _download_url(candidate: DatasetCandidate, dest_dir: Path) -> DatasetDownloadResult:
    try:
        url = candidate.url
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Handle file:// and plain local paths
        if url.startswith("file://"):
            src = Path(url[7:])
        else:
            src = Path(url)

        if src.exists():
            dest = dest_dir / src.name
            shutil.copy2(src, dest)
            return DatasetDownloadResult(success=True, local_path=str(dest))

        # HTTP download
        import httpx

        with httpx.Client(timeout=60, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()

        filename = url.split("/")[-1] or "data.csv"
        dest = dest_dir / filename
        dest.write_bytes(resp.content)
        return DatasetDownloadResult(success=True, local_path=str(dest))
    except Exception as exc:
        logger.error("URL download failed: %s", exc)
        return DatasetDownloadResult(success=False, error=str(exc))
