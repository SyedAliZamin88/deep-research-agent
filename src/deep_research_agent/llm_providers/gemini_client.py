from __future__ import annotations
import asyncio
from typing import Any
from google import genai
from google.genai import types as genai_types
from deep_research_agent.config import settings
from deep_research_agent.core.errors import ProviderNotConfiguredError
from deep_research_agent.core.interfaces import LanguageModelProvider
from deep_research_agent.utils import get_logger
from deep_research_agent.utils.rate_limiter import default_async_rate_limiter
from deep_research_agent.utils.retry import RetryPolicy

try:
    from deep_research_agent.observability.prometheus_metrics import record_llm_call
    METRICS_AVAILABLE = True
except Exception:
    METRICS_AVAILABLE = False

    def record_llm_call(*args, **kwargs):
        return None

try:
    from deep_research_agent.observability.prometheus_metrics import record_llm_call
    METRICS_AVAILABLE = True
except Exception:
    METRICS_AVAILABLE = False

    def record_llm_call(*args, **kwargs):
        return None


class GeminiProvider(LanguageModelProvider):
    """Wrapper for Google Gemini models using the google.genai SDK."""

    def __init__(
        self,
        model: str = "models/gemini-3-flash-preview",
        temperature: float = 0.2,
        top_p: float = 0.95,
    ) -> None:
        if not settings.secrets.google_api_key:
            raise ProviderNotConfiguredError("Gemini API key is not configured.")

        self.client = genai.Client(api_key=settings.secrets.google_api_key)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.retry_policy = RetryPolicy()
        self.rate_limiter = default_async_rate_limiter()
        self.logger = get_logger(__name__)

    async def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Invoke the Gemini model asynchronously with best-effort OpenInference tracing."""
        async with self.rate_limiter.async_context():
            try:
                from deep_research_agent.observability.openinference import get_default_client as _get_openinference_client  # type: ignore
                oi_client = _get_openinference_client() if _get_openinference_client else None
            except Exception:
                oi_client = None

            model_name = kwargs.get("model", self.model)

            if oi_client:
                with oi_client.trace_llm_call(provider_name="gemini", model=model_name, prompt=prompt) as trace_ctx:
                    result = await asyncio.to_thread(self._invoke_with_retry, prompt, kwargs)
                    try:
                        if isinstance(result, str):
                            trace_ctx.add_output(result)
                        elif isinstance(result, dict):
                            usage = result.get("usage") or {}
                            input_t = int(usage.get("prompt_tokens", 0) or 0)
                            output_t = int(usage.get("completion_tokens", 0) or 0)
                            if input_t or output_t:
                                trace_ctx.record_token_counts(input_tokens=input_t, output_tokens=output_t)
                            text = None
                            if "output" in result and isinstance(result.get("output"), list):
                                parts = []
                                for out in result.get("output", []) or []:
                                    if isinstance(out, dict):
                                        for part in out.get("parts", []) or []:
                                            txt = part.get("text") or part.get("content")
                                            if isinstance(txt, str):
                                                parts.append(txt)
                                if parts:
                                    text = "\n".join(parts)
                            text = text or result.get("text") or result.get("response") or None
                            if isinstance(text, str) and text:
                                trace_ctx.add_output(text)
                    except Exception:
                        pass
                    return result
            else:
                return await asyncio.to_thread(self._invoke_with_retry, prompt, kwargs)

    def _invoke_with_retry(self, prompt: str, kwargs: dict[str, Any]) -> str:
        @self.retry_policy.wrap
        def _call() -> str:
            genai_model = kwargs.get("model", self.model)
            system_prompt: str | None = kwargs.get("system_prompt")
            temperature = kwargs.get("temperature", self.temperature)
            top_p = kwargs.get("top_p", self.top_p)

            contents = self._build_contents(prompt, system_prompt, kwargs.get("context_messages"))

            response = self.client.models.generate_content(
                model=genai_model,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    temperature=temperature,
                    top_p=top_p,
                ),
            )
            return self._extract_text(response)

        return _call()

    def _build_contents(
        self,
        prompt: str,
        system_prompt: str | None,
        context_messages: list[dict[str, Any]] | None = None,
    ) -> list[genai_types.Content]:
        contents: list[genai_types.Content] = []

        if system_prompt:
            contents.append(
                genai_types.Content(
                    role="system",
                    parts=[genai_types.Part(text=system_prompt)],
                )
            )

        if context_messages:
            for message in context_messages:
                role = message.get("role", "user")
                text = message.get("text")
                if not text:
                    continue
                contents.append(
                    genai_types.Content(
                        role=role,
                        parts=[genai_types.Part(text=text)],
                    )
                )

        contents.append(
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=prompt)],
            )
        )
        return contents

    def _extract_text(self, response: genai_types.GenerateContentResponse) -> str:
        if hasattr(response, "text") and response.text:
            return response.text

        collected: list[str] = []
        for output in getattr(response, "output", []) or []:
            for part in getattr(output, "parts", []) or []:
                text = getattr(part, "text", None)
                if text:
                    collected.append(text)
        return "\n".join(collected).strip()
