"""
OpenInference observability integration for Deep Research Agent.

This module provides a thin integration layer targeting the OpenInference
style tracing/telemetry primitives used for LLM applications. It is designed
to work whether the `openinference` package (or a similarly-named SDK)
is available or not:

- When `openinference` is importable, the module initializes a tracer and
  exposes helpers to create spans, record AI-specific attributes (tokens,
  embeddings, TTFT, etc.), and emit metrics.
- When unavailable, the module falls back to a lightweight in-process
  recorder that writes structured events to the standard logging facility.

Goals / Features
- Distributed tracing helpers (span context manager / decorator).
- Standardized event/attribute names for LLM workloads (inputs, outputs,
  token counts, embeddings, latencies).
- Convenience wrappers to instrument model/provider calls (sync & async).
- Graceful fallback that keeps code paths stable even if the SDK is missing.
- Small, testable API surface suitable for unit tests or manual inspection.

Note: This implementation intentionally avoids depending on concrete exporter
code or networked telemetry backends. Exporter configuration (OTLP, Zipkin,
Prometheus, etc.) should be handled by application bootstrap code when a
real OpenInference-compatible SDK is present.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable, Coroutine

try:

    import openinference as oi
    _OPENINFERENCE_AVAILABLE = True
except Exception:
    oi = None
    _OPENINFERENCE_AVAILABLE = False

_logger = logging.getLogger("deep_research_agent.observability.openinference")


@dataclass
class TokenMetrics:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def update_from_counts(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class LatencyMetrics:
    time_request_start: float = field(default_factory=lambda: 0.0)
    time_first_token: Optional[float] = None
    time_last_token: Optional[float] = None

    def mark_start(self) -> None:
        self.time_request_start = time.monotonic()

    def mark_first_token(self) -> None:
        if self.time_first_token is None:
            self.time_first_token = time.monotonic()

    def mark_last_token(self) -> None:
        self.time_last_token = time.monotonic()

    @property
    def ttft(self) -> Optional[float]:
        if self.time_request_start and self.time_first_token:
            return max(0.0, self.time_first_token - self.time_request_start)
        return None

    @property
    def end_to_end(self) -> Optional[float]:
        if self.time_request_start and self.time_last_token:
            return max(0.0, self.time_last_token - self.time_request_start)
        return None

    def time_per_token(self) -> Optional[float]:
        if self.time_first_token and self.time_last_token:
            duration = max(1e-6, self.time_last_token - (self.time_first_token or self.time_request_start))
            return duration
        return None


class _InMemoryRecorder:
    """
    Lightweight fallback recorder used when OpenInference SDK is not present.

    It records trace-like events to the Python logger at INFO level with a
    structured dictionary payload. This keeps observability calls cheap and
    readable in local development and tests.
    """

    def __init__(self) -> None:
        self._events: list[Dict[str, Any]] = []

    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        span = {
            "name": name,
            "start": time.time(),
            "attributes": attributes or {},
            "events": [],
            "ended": False,
        }
        self._events.append(span)
        _logger.info("openinference.fallback.span_start", extra={"span": span})
        return span

    def end_span(self, span: Dict[str, Any], attributes: Optional[Dict[str, Any]] = None) -> None:
        span["end"] = time.time()
        span["ended"] = True
        if attributes:
            span["attributes"].update(attributes)
        _logger.info("openinference.fallback.span_end", extra={"span": span})

    def add_event(self, span: Dict[str, Any], event_name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        ev = {"ts": time.time(), "name": event_name, "payload": payload or {}}
        span["events"].append(ev)
        _logger.info("openinference.fallback.span_event", extra={"span": span, "event": ev})

    def record_metric(self, name: str, value: Any, tags: Optional[Dict[str, Any]] = None) -> None:
        _logger.info("openinference.fallback.metric", extra={"metric": {"name": name, "value": value, "tags": tags}})

    def get_events(self) -> list[Dict[str, Any]]:
        return list(self._events)


class OpenInferenceClient:
    """
    High-level client wrapper that exposes an opinionated API for tracing LLM
    interactions via OpenInference (if available) or via the fallback recorder.

    Typical usage:
        client = OpenInferenceClient(service_name="deep-research-agent")
        with client.span("plan_generation") as span:
            span.add_event("prompt_sent", {"prompt": "..."} )
            ...
    """

    def __init__(self, service_name: str = "deep-research-agent", exporter_url: Optional[str] = None) -> None:
        self.service_name = service_name
        self.exporter_url = exporter_url
        self._recorder = _InMemoryRecorder()
        self._active_tracer = None

        if _OPENINFERENCE_AVAILABLE:
            try:
                self._active_tracer = getattr(oi, "Tracer")(
                    service_name=service_name, exporter_url=exporter_url
                )
                _logger.info("openinference.initialized", service=service_name)
            except Exception as exc:  # pragma: no cover - defensive
                _logger.exception("openinference.init_failed", exc=exc)
                self._active_tracer = None
        else:
            _logger.debug("openinference.sdk_unavailable", service=service_name)

    @contextlib.contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Context manager for a span. Yields a thin `Span`-like helper object.

        The yielded object has methods:
         - add_event(name, payload)
         - set_attribute(k, v)
         - end()
        """
        if self._active_tracer:

            try:
                span = self._active_tracer.start_span(name, attributes=attributes or {})
            except Exception as exc:  # pragma: no cover - defensive
                _logger.exception("openinference.span_start_error", exc=exc)
                span = self._recorder.start_span(name, attributes=attributes)
            try:
                yield _OpenInferenceSpanAdapter(span, self._active_tracer)
            finally:
                try:
                    self._active_tracer.end_span(span)
                except Exception:
                    # fallback recorder end
                    if isinstance(span, dict):
                        self._recorder.end_span(span)
        else:
            span = self._recorder.start_span(name, attributes=attributes)
            try:
                yield _FallbackSpan(span, recorder=self._recorder)
            finally:
                self._recorder.end_span(span)

    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> Any:
        """Start a span and return an object suitable for adding events/attributes."""
        if self._active_tracer:
            try:
                return self._active_tracer.start_span(name, attributes=attributes or {})
            except Exception as exc:
                _logger.exception("openinference.span_start_error", exc=exc)
                return self._recorder.start_span(name, attributes=attributes)
        else:
            return self._recorder.start_span(name, attributes=attributes)

    def end_span(self, span_obj: Any, attributes: Optional[Dict[str, Any]] = None) -> None:
        if self._active_tracer:
            try:
                self._active_tracer.end_span(span_obj, attributes=attributes or {})
                return
            except Exception:
                _logger.exception("openinference.span_end_error")

        if isinstance(span_obj, dict):
            self._recorder.end_span(span_obj, attributes=attributes)

    def add_event(self, span_obj: Any, event_name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if self._active_tracer:
            try:
                span_obj.add_event(event_name, payload or {})
                return
            except Exception:
                _logger.exception("openinference.add_event_error")
        if isinstance(span_obj, dict):
            self._recorder.add_event(span_obj, event_name, payload)

    def record_metric(self, name: str, value: Any, tags: Optional[Dict[str, Any]] = None) -> None:
        if self._active_tracer:
            try:
                if hasattr(self._active_tracer, "record_metric"):
                    self._active_tracer.record_metric(name, value, tags=tags or {})
                    return
            except Exception:
                _logger.exception("openinference.metric_error")
        self._recorder.record_metric(name, value, tags=tags)

    def trace_llm_call(self, provider_name: str, model: str, prompt: str, extra: Optional[Dict[str, Any]] = None):
        """
        Returns a context manager that instruments an LLM invocation.

        Usage:
            with client.trace_llm_call("openai", "gpt-4.1", prompt) as ctx:
                # call model
                ctx.mark_first_token()
                ctx.add_output(output_text)
                ctx.record_token_counts(input_tokens=..., output_tokens=...)
        """
        extra = extra or {}
        attributes = {"provider": provider_name, "model": model}
        attributes.update(extra or {})

        class _Ctx:
            def __init__(self, outer: "OpenInferenceClient"):
                self._outer = outer
                self._span = None
                self.token_metrics = TokenMetrics()
                self.latency = LatencyMetrics()
                self._first_token_recorded = False
                self.outputs: list[str] = []
                self.inputs: list[str] = []

            def __enter__(self):
                self.latency.mark_start()
                self._span = self._outer.start_span("llm.call", attributes=attributes)
                try:
                    self._outer.add_event(self._span, "prompt.sent", {"prompt_preview": (prompt or "")[:512]})
                except Exception:
                    pass
                return self

            def __exit__(self, exc_type, exc, tb):

                self.latency.mark_last_token()
                attr: Dict[str, Any] = {
                    "input_tokens": self.token_metrics.input_tokens,
                    "output_tokens": self.token_metrics.output_tokens,
                    "total_tokens": self.token_metrics.total_tokens,
                    "ttft": self.latency.ttft,
                    "end_to_end": self.latency.end_to_end,
                    "outputs_count": len(self.outputs),
                }
                try:
                    if self.outputs:
                        attr["outputs_preview"] = [o[:1024] for o in self.outputs[:3]]
                except Exception:
                    pass

                try:
                    self._outer.record_metric("llm.token_usage", attr["total_tokens"], tags={"provider": provider_name, "model": model})
                    self._outer.add_event(self._span, "llm.summary", attr)
                except Exception:
                    _logger.exception("openinference.llm_summary_error")
                finally:
                    self._outer.end_span(self._span, attributes=attr)

            def mark_first_token(self):
                if not self._first_token_recorded:
                    self.latency.mark_first_token()
                    self._first_token_recorded = True
                    try:
                        self._outer.add_event(self._span, "first_token", {"ts": time.time()})
                    except Exception:
                        pass

            def add_output(self, text: str):
                self.outputs.append(text or "")
                try:
                    self._outer.add_event(self._span, "output.chunk", {"preview": (text or "")[:512]})
                except Exception:
                    pass

            def record_token_counts(self, input_tokens: int = 0, output_tokens: int = 0):
                self.token_metrics.update_from_counts(input_tokens=input_tokens, output_tokens=output_tokens)
                try:
                    self._outer.add_event(self._span, "token.counts", {"input": input_tokens, "output": output_tokens, "total": self.token_metrics.total_tokens})
                except Exception:
                    pass

        return _Ctx(self)

    def instrument_async_callable(self, fn: Callable[..., Coroutine], provider_name: str, model: str, prompt_arg: str = "prompt"):
        """
        Decorator factory to instrument an async callable representing an LLM invocation.

        - `fn` is expected to be an async function that takes `prompt` (or the
          configured `prompt_arg`) among its kwargs or positional args.
        - The wrapper will start an llm.call span, measure TTFT markers if the
          underlying function yields token events (best-effort), and attach
          token metrics when the coroutine returns.

        Example:
            @client.instrument_async_callable(openai_invoke, "openai", "gpt-4.1")
            async def wrapped(...):
                ...
        """
        if not asyncio.iscoroutinefunction(fn):
            raise TypeError("instrument_async_callable expects an async function")

        @functools.wraps(fn)
        async def _wrapped(*args, **kwargs):
            prompt = kwargs.get(prompt_arg)
            if prompt is None:
                if len(args) >= 1:
                    prompt = args[0]
                else:
                    prompt = ""

            with self.trace_llm_call(provider_name, model, prompt) as ctx:
                try:
                    result = await fn(*args, **kwargs)
                except Exception:

                    try:
                        self.add_event(ctx._span, "llm.exception", {"error": "exception during LLM call"})
                    except Exception:
                        pass
                    raise

                if isinstance(result, (str, bytes)):

                    ctx.add_output(result if isinstance(result, str) else result.decode("utf-8", errors="replace"))
                elif isinstance(result, dict):
                    text = result.get("text") or result.get("response") or result.get("content")
                    if isinstance(text, str):
                        ctx.add_output(text)
                    input_t = int(result.get("usage", {}).get("prompt_tokens", 0) or 0)
                    output_t = int(result.get("usage", {}).get("completion_tokens", 0) or 0)
                    ctx.record_token_counts(input_tokens=input_t, output_tokens=output_t)
                else:
                    try:
                        preview = str(result)[:1024]
                        ctx.add_output(preview)
                    except Exception:
                        pass
                return result

        return _wrapped

class _FallbackSpan:
    def __init__(self, span_dict: Dict[str, Any], recorder: _InMemoryRecorder):
        self._span = span_dict
        self._recorder = recorder

    def add_event(self, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self._recorder.add_event(self._span, name, payload)

    def set_attribute(self, key: str, value: Any) -> None:
        self._span.setdefault("attributes", {})[key] = value

    def end(self, attributes: Optional[Dict[str, Any]] = None) -> None:
        self._recorder.end_span(self._span, attributes=attributes)


class _OpenInferenceSpanAdapter:
    """
    Adapter around a real OpenInference span object to provide a consistent
    minimal API used by client code. This attempts to call the underlying
    SDK methods and degrades gracefully if they are missing.
    """

    def __init__(self, real_span: Any, tracer: Any):
        self._span = real_span
        self._tracer = tracer

    def add_event(self, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        try:
            if hasattr(self._span, "add_event"):
                self._span.add_event(name, payload or {})
            elif hasattr(self._tracer, "add_event"):
                self._tracer.add_event(self._span, name, payload or {})
            else:
                _logger.debug("openinference.span_add_event_unavailable", name=name)
        except Exception:
            _logger.exception("openinference.span_add_event_error")

    def set_attribute(self, key: str, value: Any) -> None:
        try:
            if hasattr(self._span, "set_attribute"):
                self._span.set_attribute(key, value)
            else:
                if isinstance(self._span, dict):
                    self._span.setdefault("attributes", {})[key] = value
        except Exception:
            _logger.exception("openinference.span_set_attribute_error")

    def end(self, attributes: Optional[Dict[str, Any]] = None) -> None:
        try:
            if hasattr(self._tracer, "end_span"):
                self._tracer.end_span(self._span, attributes=attributes or {})
            elif hasattr(self._span, "end"):
                self._span.end()
            else:
                _logger.debug("openinference.span_end_unavailable")
        except Exception:
            _logger.exception("openinference.span_end_error")

_default_client: Optional[OpenInferenceClient] = None


def get_default_client() -> OpenInferenceClient:
    global _default_client
    if _default_client is None:
        _default_client = OpenInferenceClient()
    return _default_client


def trace_async_fn(provider_name: str, model: str, prompt_arg: str = "prompt"):
    """
    Decorator to instrument an async function with the default OpenInference
    client. Example:

        @trace_async_fn("openai", "gpt-4.1")
        async def call_model(prompt): ...
    """
    def _decorator(fn: Callable[..., Coroutine]):
        client = get_default_client()
        return client.instrument_async_callable(fn, provider_name, model, prompt_arg)
    return _decorator

__all__ = [
    "OpenInferenceClient",
    "get_default_client",
    "trace_async_fn",
    "TokenMetrics",
    "LatencyMetrics",
]
