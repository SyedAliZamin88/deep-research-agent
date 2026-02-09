from typing import Literal
from .query_planner import QueryPlanner
from .tavily_client import TavilyProvider

SearchProviderType = Literal["tavily", "web"]

def create_search_provider(provider: SearchProviderType, **kwargs):
    """Factory function to create search providers."""
    from deep_research_agent.core.errors import ProviderNotConfiguredError

    if provider == "tavily":
        return TavilyProvider(**kwargs)

    if provider == "web":
        from .web_scraper import WebScraperProvider
        return WebScraperProvider(**kwargs)

    raise ProviderNotConfiguredError(f"Unsupported search provider: {provider}")

__all__ = [
    "QueryPlanner",
    "TavilyProvider",
    "SearchProviderType",
    "create_search_provider",
]
