from __future__ import annotations
import asyncio
from typing import Any
from deep_research_agent.config import settings
from deep_research_agent.core.errors import ProviderNotConfiguredError, SearchQueryError
from deep_research_agent.core.interfaces import SearchProvider
from deep_research_agent.utils import get_logger
from deep_research_agent.utils.rate_limiter import default_async_rate_limiter
from deep_research_agent.utils.retry import RetryPolicy


class SerpApiProvider(SearchProvider):
    """SerpAPI implementation of the SearchProvider interface.

    The dependency on the official SerpAPI client (`google-search-results`)
    is imported lazily so that the project can operate without the optional
    package when SerpAPI is not being used.
    """

    def __init__(self) -> None:
        if not settings.secrets.serpapi_api_key:
            raise ProviderNotConfiguredError("SerpAPI key is not configured. Set SERPAPI_API_KEY in your environment.")
        try:
            from google_search_results import GoogleSearchResults

        except ImportError:
            try:
                from serpapi import GoogleSearch
            except ImportError as exc:
                raise ProviderNotConfiguredError(

                    "SerpAPI client library is not installed. Install the optional 'serp' dependency group."

                ) from exc

            class _SerpApiShim:
                def __init__(self, params):
                    self._params = dict(params)

                @property
                def params(self):
                    return self._params

                def get_dict(self):
                    search = GoogleSearch(self._params)
                    return search.get_dict()

            GoogleSearchResults = _SerpApiShim

        self._client_cls = GoogleSearchResults
        self._client = self._client_cls({"api_key": settings.secrets.serpapi_api_key})
        self.logger = get_logger(__name__)
        self.retry_policy = RetryPolicy()
        self.rate_limiter = default_async_rate_limiter()

    async def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Perform a SerpAPI search asynchronously."""
        async with self.rate_limiter.async_context():
            return await asyncio.to_thread(self._search_sync, query, kwargs)

    def _search_sync(self, query: str, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        """Blocking SerpAPI call executed inside a retry wrapper."""

        @self.retry_policy.wrap
        def _invoke() -> list[dict[str, Any]]:
            params = {
                "q": query,
                "num": kwargs.get("num_results", 10),
                "engine": "google",
                "gl": kwargs.get("gl", "us"),
                "hl": kwargs.get("hl", "en"),
            }

            self._client.params.update(params)
            result = self._client.get_dict()

            organic_results = result.get("organic_results")
            if not organic_results:
                raise SearchQueryError(f"SerpAPI returned no organic results for query: {query}")

            records: list[dict[str, Any]] = []
            for item in organic_results:
                records.append(
                    {
                        "title": item.get("title"),
                        "url": item.get("link"),
                        "snippet": item.get("snippet"),
                        "source": "serpapi",
                        "metadata": item,
                    }
                )
            return records

        try:
            records = _invoke()
            self.logger.info("serpapi.search.success", query=query, results=len(records))
            return records
        except Exception as exc:
            self.logger.warning("serpapi.search.failure", query=query, error=str(exc))
            raise
