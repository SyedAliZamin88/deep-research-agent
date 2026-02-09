from __future__ import annotations
from typing import Any
from tavily import TavilyClient
from deep_research_agent.config import settings
from deep_research_agent.core.errors import ProviderNotConfiguredError, SearchQueryError
from deep_research_agent.core.interfaces import SearchProvider
from deep_research_agent.utils import get_logger
from deep_research_agent.utils.rate_limiter import default_async_rate_limiter
from deep_research_agent.utils.retry import RetryPolicy

class TavilyProvider(SearchProvider):
    """Tavily search API integration."""

    def __init__(self) -> None:
        if not settings.secrets.tavily_api_key:
            raise ProviderNotConfiguredError("Tavily API key is not configured.")

        self.client = TavilyClient(api_key=settings.secrets.tavily_api_key)
        self.retry_policy = RetryPolicy()
        self.rate_limiter = default_async_rate_limiter()
        self.logger = get_logger(__name__)

    async def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Execute search query via Tavily API."""
        async with self.rate_limiter.async_context():
            return await self._search_with_retry(query, **kwargs)

    async def _search_with_retry(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        @self.retry_policy.wrap
        def _call() -> list[dict[str, Any]]:
            try:
                response = self.client.search(
                    query=query,
                    search_depth=kwargs.get("search_depth", "advanced"),
                    max_results=kwargs.get("num_results", 10),
                )
            except Exception as e:
                raise SearchQueryError(f"Tavily search failed: {str(e)}")

            if "results" not in response:
                return []

            return [
                {
                    "title": entry.get("title", "Untitled"),
                    "url": entry.get("url", ""),
                    "snippet": entry.get("content", ""),
                    "source": "tavily",
                    "metadata": {"score": entry.get("score"), "published_date": entry.get("published_date")},
                }
                for entry in response.get("results", [])
            ]

        return _call()
