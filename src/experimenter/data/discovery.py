"""Dataset discovery: Kaggle, HuggingFace Hub, and public web."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import httpx
from bs4 import BeautifulSoup

from experimenter.models import DatasetCandidate

logger = logging.getLogger(__name__)


async def search_kaggle(
    query: str, limit: int = 5, _api: Any = None
) -> list[DatasetCandidate]:
    """Search Kaggle datasets. Returns [] on any failure.

    _api: injected KaggleApiExtended instance (for testing).
    """
    try:
        if _api is None:
            import kaggle.api.kaggle_api_extended as _m  # noqa: PLC0415

            _api = _m.KaggleApiExtended()
            _api.authenticate()
        datasets = _api.dataset_list(search=query, sort_by="relevance", page_size=limit)
        results: list[DatasetCandidate] = []
        for ds in list(datasets)[:limit]:
            results.append(
                DatasetCandidate(
                    id=str(ds.ref),
                    name=str(ds.title),
                    source="kaggle",
                    description=str(ds.subtitle or ""),
                    url=f"https://www.kaggle.com/datasets/{ds.ref}",
                    size_mb=(ds.totalBytes / (1024 * 1024)) if ds.totalBytes else None,
                    relevance_score=0.8,
                )
            )
        return results
    except Exception as exc:
        logger.warning("Kaggle search failed: %s", exc)
        return []


async def search_huggingface(
    query: str, limit: int = 5, _api: Any = None
) -> list[DatasetCandidate]:
    """Search HuggingFace Hub datasets. Returns [] on any failure.

    _api: injected HfApi instance (for testing).
    """
    try:
        if _api is None:
            from huggingface_hub import HfApi  # noqa: PLC0415

            _api = HfApi()
        datasets = list(_api.list_datasets(search=query, limit=limit))
        results: list[DatasetCandidate] = []
        for ds in datasets[:limit]:
            results.append(
                DatasetCandidate(
                    id=str(ds.id),
                    name=str(ds.id),
                    source="huggingface",
                    description=str(getattr(ds, "description", "") or ""),
                    url=f"https://huggingface.co/datasets/{ds.id}",
                    relevance_score=0.7,
                )
            )
        return results
    except Exception as exc:
        logger.warning("HuggingFace search failed: %s", exc)
        return []


async def search_web(query: str, limit: int = 5) -> list[DatasetCandidate]:
    """Scrape DuckDuckGo HTML for public dataset links. Returns [] on failure."""
    try:
        search_query = f"{query} dataset CSV download"
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": search_query},
                headers={"User-Agent": "Mozilla/5.0 (compatible; experimenter/0.1)"},
            )
        soup = BeautifulSoup(resp.text, "html.parser")
        results: list[DatasetCandidate] = []
        for result in soup.select(".result__title")[:limit]:
            link = result.find("a")
            if link and link.get("href"):
                href = str(link["href"])
                results.append(
                    DatasetCandidate(
                        id=href,
                        name=link.get_text(strip=True) or href,
                        source="web",
                        description="",
                        url=href,
                        relevance_score=0.5,
                    )
                )
        return results
    except Exception as exc:
        logger.warning("Web search failed: %s", exc)
        return []


async def search_all(
    query: str,
    sources: Optional[list[str]] = None,
    limit_per_source: int = 5,
) -> list[DatasetCandidate]:
    """Search all configured sources and merge results, sorted by relevance."""
    if sources is None:
        sources = ["kaggle", "huggingface", "web"]

    tasks: list = []
    if "kaggle" in sources:
        tasks.append(search_kaggle(query, limit_per_source))
    if "huggingface" in sources:
        tasks.append(search_huggingface(query, limit_per_source))
    if "web" in sources:
        tasks.append(search_web(query, limit_per_source))

    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    candidates: list[DatasetCandidate] = []
    for r in all_results:
        if isinstance(r, list):
            candidates.extend(r)
        elif isinstance(r, Exception):
            logger.warning("Source search raised: %s", r)

    candidates.sort(key=lambda c: c.relevance_score, reverse=True)
    return candidates
