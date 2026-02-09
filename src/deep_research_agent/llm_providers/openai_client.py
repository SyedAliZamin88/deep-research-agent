from __future__ import annotations

from typing import Any

import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from deep_research_agent.config import settings
from deep_research_agent.core.errors import ProviderNotConfiguredError
from deep_research_agent.core.interfaces import LanguageModelProvider
from deep_research_agent.utils import get_logger
from deep_research_agent.utils.rate_limiter import default_async_rate_limiter
from deep_research_agent.utils.retry import RetryPolicy

try:
    from deep_research_agent.observability.openinference import get_default_client as _get_openinference_client  # type: ignore
except Exception:
    _get_openinference_client = None

try:
    from deep_research_agent.observability.prometheus_metrics import record_llm_call as _record_llm_call
    METRICS_AVAILABLE = True
except Exception:
    METRICS_AVAILABLE = False

    def _record_llm_call(*args, **kwargs):
        return None


class OpenAIProvider(LanguageModelProvider):
    """Wrapper around OpenAI Chat Completions API."""

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> None:
        if not settings.secrets.openai_api_key:
            raise ProviderNotConfiguredError("OpenAI API key is not configured.")
        self.client = OpenAI(api_key=settings.secrets.openai_api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_policy = RetryPolicy()
        self.rate_limiter = default_async_rate_limiter()
        self.logger = get_logger(__name__)

    async def invoke(self, prompt: str, **kwargs: Any) -> str:
        """
        Invoke the OpenAI model with rate limiting, retry, and telemetry.

        Behavior:
        - Attempts to wrap call in OpenInference trace span if available.
        - Always records telemetry metrics using prometheus_metrics helper.
        """
        import time

        async with self.rate_limiter.async_context():
            oi_client = None
            try:
                if _get_openinference_client:
                    oi_client = _get_openinference_client()
            except Exception:
                oi_client = None

            model_name = kwargs.get("model", self.model)

            start_ts = time.monotonic()
            result = await self._invoke_with_retry(prompt, **kwargs)
            elapsed = max(0.0, time.monotonic() - start_ts)

            input_t = 0
            output_t = 0
            try:
                if isinstance(result, dict):
                    usage = result.get("usage") or {}
                    input_t = int(usage.get("prompt_tokens", 0) or 0)
                    output_t = int(usage.get("completion_tokens", 0) or 0)
                elif isinstance(result, str):
                    pass

                if METRICS_AVAILABLE:
                    try:
                        _record_llm_call(
                            provider="openai",
                            input_tokens=input_t,
                            output_tokens=output_t,
                            latency_s=elapsed,
                        )
                    except Exception:
                        pass
            except Exception:
                pass

            if oi_client:
                try:
                    with oi_client.trace_llm_call(provider_name="openai", model=model_name, prompt=prompt) as trace_ctx:

                        try:
                            if isinstance(result, str):
                                trace_ctx.add_output(result)
                            elif isinstance(result, dict):
                                text = None
                                if result.get("choices") and len(result.get("choices")):
                                    text = result.get("choices")[0].get("message", {}).get("content")
                                else:
                                    text = result.get("text") or result.get("response")
                                if text:
                                    trace_ctx.add_output(text)
                                if input_t or output_t:
                                    trace_ctx.record_token_counts(input_tokens=input_t, output_tokens=output_t)
                        except Exception:
                            pass
                except Exception:
                    pass

            return result

    async def _invoke_with_retry(self, prompt: str, **kwargs: Any) -> str:
        @self.retry_policy.wrap
        def _call() -> str:
            messages: list[ChatCompletionMessageParam] = [
                {"role": "system", "content": kwargs.get("system_prompt", "")},
                {"role": "user", "content": prompt},
            ]
            response = self.client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                timeout=settings.runtime.request_timeout_seconds,
            )
            return response.choices[0].message.content or ""

        return _call()
