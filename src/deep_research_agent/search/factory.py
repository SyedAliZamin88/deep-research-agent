from __future__ import annotations
from typing import Literal
from deep_research_agent.core.errors import ProviderNotConfiguredError
from deep_research_agent.core.interfaces import SearchProvider

SearchProviderType = Literal["tavily", "serpapi", "web", "openai_websearch", "gemini_websearch"]

def create_search_provider(provider: SearchProviderType) -> SearchProvider:
    """Instantiate a search provider, handling optional dependencies gracefully."""
    if provider == "tavily":
        from deep_research_agent.search.tavily_client import TavilyProvider

        return TavilyProvider()

    if provider == "serpapi":
        return _create_serp_provider()

    if provider == "openai_websearch":
        from deep_research_agent.search.openai_websearch import OpenAIWebSearchProvider
        return OpenAIWebSearchProvider()

    if provider == "gemini_websearch":
        from deep_research_agent.search.gemini_websearch import GeminiWebSearchProvider
        return GeminiWebSearchProvider()

    if provider == "web":
        from deep_research_agent.search.web_scraper import WebScraperProvider

        return WebScraperProvider()

    raise ProviderNotConfiguredError(f"Unsupported search provider requested: {provider}")


def _create_serp_provider() -> SearchProvider:
    """Instantiate the SerpAPI provider, surfacing helpful guidance when unavailable."""
    try:
        from deep_research_agent.search.serp_client import SerpApiProvider
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ProviderNotConfiguredError(
            "SerpAPI provider is unavailable because the optional dependency "
            "'google-search-results' is not installed. Install it with "
            "`python -m uv add google-search-results` or enable the 'serp' extra."
        ) from exc

    try:
        return SerpApiProvider()
    except ProviderNotConfiguredError as exc:
        raise ProviderNotConfiguredError(
            "SerpAPI provider is not configured. Ensure SERPAPI_API_KEY is set or "
            "choose a different search provider."
        ) from exc
