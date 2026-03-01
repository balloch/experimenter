"""Tests for dataset discovery (Kaggle, HuggingFace, web)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from experimenter.data.discovery import (
    search_all,
    search_huggingface,
    search_kaggle,
    search_web,
)
from experimenter.models import DatasetCandidate


# Use dependency injection (_api param) to avoid triggering kaggle's auth-on-import.

class TestSearchKaggle:
    async def test_returns_dataset_candidates(self):
        mock_api = MagicMock()
        mock_ds = MagicMock()
        mock_ds.ref = "user/exercise-gpa"
        mock_ds.title = "Exercise & GPA"
        mock_ds.subtitle = "Student data"
        mock_ds.totalBytes = 2_000_000
        mock_api.dataset_list.return_value = [mock_ds]

        results = await search_kaggle("exercise gpa", limit=5, _api=mock_api)

        assert len(results) == 1
        assert isinstance(results[0], DatasetCandidate)
        assert results[0].source == "kaggle"
        assert results[0].id == "user/exercise-gpa"
        assert results[0].size_mb == pytest.approx(2_000_000 / (1024 * 1024), rel=0.01)

    async def test_returns_empty_on_api_failure(self):
        mock_api = MagicMock()
        mock_api.dataset_list.side_effect = Exception("No credentials")

        results = await search_kaggle("anything", _api=mock_api)

        assert results == []

    async def test_respects_limit(self):
        mock_api = MagicMock()
        mock_ds = MagicMock()
        mock_ds.ref = "user/ds"
        mock_ds.title = "DS"
        mock_ds.subtitle = ""
        mock_ds.totalBytes = 0
        mock_api.dataset_list.return_value = [mock_ds] * 10

        results = await search_kaggle("q", limit=3, _api=mock_api)

        assert len(results) <= 3


class TestSearchHuggingFace:
    async def test_returns_dataset_candidates(self):
        mock_api = MagicMock()
        mock_ds = MagicMock()
        mock_ds.id = "mldb/student-performance"
        mock_ds.description = "Student records"
        mock_api.list_datasets.return_value = iter([mock_ds])

        results = await search_huggingface("student performance", limit=5, _api=mock_api)

        assert len(results) == 1
        assert results[0].source == "huggingface"
        assert results[0].id == "mldb/student-performance"
        assert "huggingface.co" in results[0].url

    async def test_returns_empty_on_failure(self):
        mock_api = MagicMock()
        mock_api.list_datasets.side_effect = Exception("Network error")

        results = await search_huggingface("anything", _api=mock_api)

        assert results == []


class TestSearchWeb:
    @patch("experimenter.data.discovery.httpx.AsyncClient")
    async def test_returns_dataset_candidates(self, MockClient):
        html = """
        <html><body>
        <div class="result__title"><a href="https://github.com/example/data">Example Dataset</a></div>
        </body></html>
        """
        mock_resp = MagicMock()
        mock_resp.text = html
        MockClient.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_resp)

        results = await search_web("exercise gpa dataset", limit=5)

        assert isinstance(results, list)
        for r in results:
            assert r.source == "web"

    @patch("experimenter.data.discovery.httpx.AsyncClient")
    async def test_returns_empty_on_network_failure(self, MockClient):
        MockClient.return_value.__aenter__.return_value.get = AsyncMock(
            side_effect=Exception("timeout")
        )

        results = await search_web("anything")

        assert results == []


class TestSearchAll:
    @patch("experimenter.data.discovery.search_kaggle", new_callable=AsyncMock)
    @patch("experimenter.data.discovery.search_huggingface", new_callable=AsyncMock)
    @patch("experimenter.data.discovery.search_web", new_callable=AsyncMock)
    async def test_combines_results_from_all_sources(
        self, mock_web, mock_hf, mock_kaggle, sample_dataset_candidate
    ):
        mock_kaggle.return_value = [sample_dataset_candidate]
        mock_hf.return_value = [
            DatasetCandidate(
                id="hf/ds", name="HF DS", source="huggingface",
                description="", url="http://hf", relevance_score=0.7
            )
        ]
        mock_web.return_value = []

        results = await search_all("student exercise gpa")

        assert len(results) == 2
        assert mock_kaggle.called
        assert mock_hf.called
        assert mock_web.called

    @patch("experimenter.data.discovery.search_kaggle", new_callable=AsyncMock)
    @patch("experimenter.data.discovery.search_huggingface", new_callable=AsyncMock)
    @patch("experimenter.data.discovery.search_web", new_callable=AsyncMock)
    async def test_sorts_by_relevance_descending(self, mock_web, mock_hf, mock_kaggle):
        low = DatasetCandidate(
            id="a", name="a", source="web", description="", url="u", relevance_score=0.3
        )
        high = DatasetCandidate(
            id="b", name="b", source="kaggle", description="", url="u", relevance_score=0.9
        )
        mock_kaggle.return_value = [low]
        mock_hf.return_value = [high]
        mock_web.return_value = []

        results = await search_all("q")

        assert results[0].relevance_score >= results[-1].relevance_score

    @patch("experimenter.data.discovery.search_kaggle", new_callable=AsyncMock)
    @patch("experimenter.data.discovery.search_huggingface", new_callable=AsyncMock)
    @patch("experimenter.data.discovery.search_web", new_callable=AsyncMock)
    async def test_respects_sources_filter(self, mock_web, mock_hf, mock_kaggle):
        mock_kaggle.return_value = []
        mock_hf.return_value = []
        mock_web.return_value = []

        await search_all("q", sources=["kaggle"])

        mock_kaggle.assert_called_once()
        mock_hf.assert_not_called()
        mock_web.assert_not_called()

    @patch("experimenter.data.discovery.search_kaggle", new_callable=AsyncMock)
    @patch("experimenter.data.discovery.search_huggingface", new_callable=AsyncMock)
    @patch("experimenter.data.discovery.search_web", new_callable=AsyncMock)
    async def test_handles_partial_source_failure(self, mock_web, mock_hf, mock_kaggle):
        """If one source fails, others still return results."""
        mock_kaggle.side_effect = Exception("Kaggle down")
        mock_hf.return_value = [
            DatasetCandidate(id="x", name="x", source="huggingface", description="", url="u")
        ]
        mock_web.return_value = []

        results = await search_all("q")
        assert isinstance(results, list)
