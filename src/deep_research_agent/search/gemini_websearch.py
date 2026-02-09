from __future__ import annotations

from typing import Any, List
from deep_research_agent.core.interfaces import SearchProvider
from deep_research_agent.config import settings

from google import genai
from google.genai import types


class GeminiWebSearchProvider(SearchProvider):
    """Search provider that uses Google Gemini web search tool."""

    def __init__(self) -> None:
        if not settings.secrets.google_api_key:
            raise RuntimeError("Google API key is not configured for Gemini web search.")
        self.client = genai.Client()

    async def search(self, query: str, num_results: int = 10, **kwargs: Any) -> List[dict[str, Any]]:
        try:
            grounding_tool = types.Tool(
                google_search=types.GoogleSearch()
            )

            config = types.GenerateContentConfig(
                tools=[grounding_tool]
            )

            response = self.client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=query,
                config=config,
            )

            text_output = response.text or ""
            return [{"title": query, "url": "", "snippet": text_output}]
        except Exception:
            return []
