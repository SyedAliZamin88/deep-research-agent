from __future__ import annotations

from typing import Any, List, Dict

from deep_research_agent.core.interfaces import SearchProvider
from deep_research_agent.config import settings
from openai import OpenAI


class OpenAIWebSearchProvider(SearchProvider):
    """Search provider that uses OpenAI's web search tool in GPT models."""

    def __init__(self) -> None:
        if not settings.secrets.openai_api_key:
            raise RuntimeError("OpenAI API key is not configured for web search.")
        self.client = OpenAI(api_key=settings.secrets.openai_api_key)

    async def search(self, query: str, num_results: int = 10, **kwargs: Any) -> List[Dict[str, Any]]:
        try:
            response = await self.client.responses.create(
                model="gpt-5",
                tools=[{"type": "web_search"}],
                input=query,
                **kwargs
            )
            output = response.output_text or ""

            return [{"title": query, "url": "", "snippet": output}]
        except Exception as e:
            return []
