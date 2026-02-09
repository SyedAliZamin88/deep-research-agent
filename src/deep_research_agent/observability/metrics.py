"""
OpenTelemetry metrics helper for Deep Research Agent.

- Best-effort OpenTelemetry metrics instrumentation: if OpenTelemetry metrics
  API is available, we create a `Meter` and a small set of instruments:
    - llm_calls (counter)
    - tokens_input (counter)
    - tokens_output (counter)
    - tokens_total (counter)
    - ttft_ms (histogram)
    - llm_latency_ms (histogram)
    - cost_usd (counter)

- If OpenTelemetry is not installed, a lightweight in-memory recorder is used
  which logs metric updates and retains simple aggregated values for debugging
  or export via `get_metrics_snapshot()`.

Usage:
    from deep_research_agent.observability.metrics import (
        record_llm_call,
        record_token_counts,
        record_ttft,
        record_latency,
        record_cost,
        get_metrics_snapshot,
    )

Design goals:
- Non-fatal: metrics calls should never crash the main application.
- Minimal friction: functions are simple convenience wrappers used across the
  codebase (LLM providers, orchestrator, etc.).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Dict, Optional

_logger = logging.getLogger("deep_research_agent.observability.metrics")
_logger.addHandler(logging.NullHandler())

_DEFAULT_COSTS = {
    "openai": float(os.getenv("COST_PER_1K_OPENAI", "0.03")),
    "gemini": float(os.getenv("COST_PER_1K_GEMINI", "0.03")),
    "ollama": float(os.getenv("COST_PER_1K_OLLAMA", "0.0")),
    "default": float(os.getenv("COST_PER_1K_DEFAULT", "0.03")),
}

_METRICS_AVAILABLE = False
_meter = None
_llm_calls = None
_tokens_input = None
_tokens_output = None
_tokens_total = None
_ttft_hist = None
_latency_hist = None
_cost_counter = None

try:

    from opentelemetry import metrics as _otel_metrics
    from opentelemetry.sdk.metrics import MeterProvider as _MeterProvider
    from opentelemetry.sdk.resources import Resource as _Resource
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader as _PeriodicReader

    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter as _OTLPMetricExporter,
        )
        _have_otlp_metric_exporter = True
    except Exception:
        _OTLPMetricExporter = None
        _have_otlp_metric_exporter = False

    _resource = _Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "deep-research-agent")})
    _provider = _MeterProvider(resource=_resource)

    if _have_otlp_metric_exporter and _OTLPMetricExporter is not None:
        try:
            _endpoint = os.getenv("OTEL_METRIC_EXPORTER_ENDPOINT", "http://localhost:4318/v1/metrics")
            _exporter = _OTLPMetricExporter(endpoint=_endpoint)
            _reader = _PeriodicReader(_exporter)
            _provider._sdk_config.metric_readers.append(_reader)
        except Exception:
            _logger.debug("Failed to attach OTLP metric exporter, continuing without it", exc_info=True)

    _otel_metrics.set_meter_provider(_provider)
    _meter = _otel_metrics.get_meter("deep-research-agent")

    # Create instruments
    _llm_calls = _meter.create_counter(
        name="llm_calls_total",
        description="Count of LLM calls made by the agent",
    )
    _tokens_input = _meter.create_counter(
        name="llm_input_tokens_total",
        description="Total input tokens sent to LLM providers",
    )
    _tokens_output = _meter.create_counter(
        name="llm_output_tokens_total",
        description="Total output tokens produced by LLM providers",
    )
    _tokens_total = _meter.create_counter(
        name="llm_tokens_total",
        description="Total tokens (input+output) for LLM usage",
    )
    _ttft_hist = _meter.create_histogram(
        name="llm_ttft_ms",
        description="Time to first token (ms) for LLM responses",
    )
    _latency_hist = _meter.create_histogram(
        name="llm_latency_ms",
        description="End-to-end latency for LLM responses (ms)",
    )
    _cost_counter = _meter.create_counter(
        name="llm_cost_usd_total",
        description="Accumulated estimated cost for LLM usage in USD",
    )

    _METRICS_AVAILABLE = True
    _logger.debug("OpenTelemetry metrics initialized for Deep Research Agent")
except Exception:
    _METRICS_AVAILABLE = False
    _logger.debug("OpenTelemetry metrics not available; using in-memory fallback")


class _InMemoryMetrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.counters: Dict[str, float] = {}
        self.histograms: Dict[str, list] = {}

    def add(self, name: str, amount: float = 1.0, attrs: Optional[Dict] = None):
        with self.lock:
            self.counters[name] = self.counters.get(name, 0.0) + float(amount)
        _logger.info("metric.increment", metric=name, amount=amount, attributes=attrs)

    def record(self, name: str, value: float, attrs: Optional[Dict] = None):
        with self.lock:
            self.histograms.setdefault(name, []).append(float(value))
        _logger.info("metric.record", metric=name, value=value, attributes=attrs)

    def snapshot(self):
        with self.lock:
            hist_summary = {}
            for k, vals in self.histograms.items():
                if not vals:
                    hist_summary[k] = {"count": 0}
                else:
                    hist_summary[k] = {
                        "count": len(vals),
                        "min": min(vals),
                        "max": max(vals),
                        "avg": sum(vals) / len(vals),
                    }
            return {"counters": dict(self.counters), "histograms": hist_summary}


_fallback_metrics = _InMemoryMetrics()

def _get_cost_per_1k(provider: str) -> float:
    return _DEFAULT_COSTS.get(provider, _DEFAULT_COSTS.get("default", 0.03))


def record_llm_call(provider: str = "default", attributes: Optional[Dict] = None):
    """
    Increment the LLM calls counter for a provider.
    """
    attrs = attributes or {"provider": provider}
    if _METRICS_AVAILABLE and _llm_calls is not None:
        try:
            _llm_calls.add(1, attributes=attrs)
            return
        except Exception:
            _logger.debug("Failed to record OTEL llm_calls", exc_info=True)

    _fallback_metrics.add(f"llm_calls_total.{provider}", 1.0, attrs)


def record_token_counts(provider: str = "default", input_tokens: int = 0, output_tokens: int = 0, attributes: Optional[Dict] = None):
    """
    Record token counters and update cost estimate.
    """
    attrs = attributes or {"provider": provider}
    total = int(input_tokens) + int(output_tokens)

    if _METRICS_AVAILABLE:
        try:
            if _tokens_input is not None:
                _tokens_input.add(input_tokens, attributes=attrs)
            if _tokens_output is not None:
                _tokens_output.add(output_tokens, attributes=attrs)
            if _tokens_total is not None:
                _tokens_total.add(total, attributes=attrs)
        except Exception:
            _logger.debug("Failed to record OTEL token counts", exc_info=True)
    else:
        _fallback_metrics.add(f"tokens_input_total.{provider}", input_tokens, attrs)
        _fallback_metrics.add(f"tokens_output_total.{provider}", output_tokens, attrs)
        _fallback_metrics.add(f"tokens_total.{provider}", total, attrs)

    try:
        cost_per_1k = _get_cost_per_1k(provider)
        estimated = (total / 1000.0) * cost_per_1k
        record_cost(provider, estimated, attributes=attrs)
    except Exception:
        _logger.debug("Failed to estimate LLM cost", exc_info=True)


def record_ttft(provider: str = "default", ttft_seconds: float = 0.0, attributes: Optional[Dict] = None):
    """
    Record Time To First Token (seconds) as ms in a histogram.
    """
    attrs = attributes or {"provider": provider}
    ms = float(ttft_seconds) * 1000.0
    if _METRICS_AVAILABLE and _ttft_hist is not None:
        try:
            _ttft_hist.record(ms, attributes=attrs)
            return
        except Exception:
            _logger.debug("Failed to record OTEL ttft", exc_info=True)
    _fallback_metrics.record(f"llm_ttft_ms.{provider}", ms, attrs)


def record_latency(provider: str = "default", latency_seconds: float = 0.0, attributes: Optional[Dict] = None):
    """
    Record end-to-end latency (seconds) as ms in a histogram.
    """
    attrs = attributes or {"provider": provider}
    ms = float(latency_seconds) * 1000.0
    if _METRICS_AVAILABLE and _latency_hist is not None:
        try:
            _latency_hist.record(ms, attributes=attrs)
            return
        except Exception:
            _logger.debug("Failed to record OTEL latency", exc_info=True)
    _fallback_metrics.record(f"llm_latency_ms.{provider}", ms, attrs)


def record_cost(provider: str = "default", usd_amount: float = 0.0, attributes: Optional[Dict] = None):
    """
    Record estimated cost in USD for billing approximation.
    """
    attrs = attributes or {"provider": provider}
    if _METRICS_AVAILABLE and _cost_counter is not None:
        try:
            _cost_counter.add(usd_amount, attributes=attrs)
            return
        except Exception:
            _logger.debug("Failed to record OTEL cost", exc_info=True)
    _fallback_metrics.add(f"llm_cost_usd_total.{provider}", usd_amount, attrs)


def get_metrics_snapshot() -> Dict:
    """
    Return a dictionary snapshot of current metrics (fallback or minimal OTEL read).
    """
    if _METRICS_AVAILABLE:
        pass
