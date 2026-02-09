from __future__ import annotations
from typing import Any
import httpx
from deep_research_agent.config import settings
from deep_research_agent.core.interfaces import LanguageModelProvider
from deep_research_agent.utils import get_logger
from deep_research_agent.utils.rate_limiter import default_async_rate_limiter
from deep_research_agent.utils.retry import RetryPolicy

class OllamaProvider(LanguageModelProvider):
    """Wrapper for local Ollama models (open-source)."""

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.logger = get_logger(__name__)
        if settings.secrets.llama_model_path is None:
            self.logger.info(
                "ollama.model_path_missing",
                message="LLAMA_MODEL_PATH is not configured; proceeding with default Ollama setup.",
            )
        self.retry_policy = RetryPolicy()
        self.rate_limiter = default_async_rate_limiter()
        self.request_timeout = settings.runtime.request_timeout_seconds


    async def invoke(self, prompt: str, **kwargs: Any) -> str:
        async with self.rate_limiter.async_context():
            return await self._invoke_with_retry(prompt, **kwargs)

    async def _invoke_with_retry(self, prompt: str, **kwargs: Any) -> str:
        @self.retry_policy.wrap
        def _call() -> str:
            payload = {
                "model": kwargs.get("model", self.model),
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.temperature),
                },
            }

            response = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.request_timeout,
            )

            response.raise_for_status()
            data = response.json()
            return data.get("response", "")

        return _call()


    async def aclose(self) -> None:
        """No persistent client to close."""
        return None
