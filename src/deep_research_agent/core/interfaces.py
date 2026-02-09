from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

class AgentNode(ABC):
    """Base interface for LangGraph nodes."""

    @abstractmethod
    def run(self, state: Any) -> Any:
        """Execute node logic and return updated state."""
        pass

class LanguageModelProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    async def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Invoke the model with a prompt and return text output."""
        pass

class SearchProvider(ABC):
    """Abstract base for search providers."""

    @abstractmethod
    async def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Run a search query and return structured results."""
        pass
