from __future__ import annotations
from typing import Any

from deep_research_agent.core.interfaces import SearchProvider
from deep_research_agent.utils import get_logger

class WebScraperProvider(SearchProvider):
    """Fallback web scraper when no API keys are available."""

    def __init__(self) -> None:
        self.logger = get_logger(__name__)
        self.logger.warning("web_scraper.init", message="Using fallback web scraper - limited functionality")

    async def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Minimal search implementation - returns mock results."""
        self.logger.warning("web_scraper.search", query=query, message="No real search performed")

        return [
            {
                "title": f"Mock result for: {query}",
                "url": "https://example.com",
                "snippet": "Web scraper fallback - configure Tavily or SerpAPI for real results",
                "source": "web_scraper_mock",
                "metadata": {},
            }
        ]
