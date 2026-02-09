from __future__ import annotations
from typing import Literal
from deep_research_agent.core.errors import ProviderNotConfiguredError
from deep_research_agent.core.interfaces import LanguageModelProvider
from deep_research_agent.llm_providers.gemini_client import GeminiProvider
from deep_research_agent.llm_providers.ollama_client import OllamaProvider
from deep_research_agent.llm_providers.openai_client import OpenAIProvider

ProviderType = Literal["openai", "gemini", "ollama"]

def create_provider(provider: ProviderType, **kwargs) -> LanguageModelProvider:
    if provider == "openai":
        return OpenAIProvider(**kwargs)
    if provider == "gemini":
        return GeminiProvider(**kwargs)
    if provider == "ollama":
        return OllamaProvider(**kwargs)
    raise ProviderNotConfiguredError(f"Unsupported provider type: {provider}")
