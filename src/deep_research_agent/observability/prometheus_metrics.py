from __future__ import annotations
import os
import threading
from functools import wraps
from typing import Optional
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CollectorRegistry,
    CONTENT_TYPE_LATEST,
)
from prometheus_client import REGISTRY as DEFAULT_REGISTRY

try:
    from fastapi import FastAPI, Response
    FASTAPI_AVAILABLE = True
except Exception:
    FASTAPI_AVAILABLE = False

_COST_MAP = {
    "openai": float(os.getenv("COST_PER_1K_OPENAI", "0.03")),
    "gemini": float(os.getenv("COST_PER_1K_GEMINI", "0.03")),
    "ollama": float(os.getenv("COST_PER_1K_OLLAMA", "0.0")),
    "default": float(os.getenv("COST_PER_1K_DEFAULT", "0.03")),
}

_METRICS_REGISTRY: CollectorRegistry = DEFAULT_REGISTRY
_metrics_lock = threading.Lock()

LLM_CALLS = Counter("llm_calls_total", "Total number of LLM calls made by provider", ["provider"], registry=_METRICS_REGISTRY)
INPUT_TOKENS = Counter("llm_input_tokens_total", "Total number of input tokens sent to LLM providers", ["provider"], registry=_METRICS_REGISTRY)
OUTPUT_TOKENS = Counter("llm_output_tokens_total", "Total number of output tokens produced by LLM providers", ["provider"], registry=_METRICS_REGISTRY)
TOKENS_TOTAL = Counter("llm_tokens_total", "Total number of tokens (in+out) for LLM providers", ["provider"], registry=_METRICS_REGISTRY)
LLM_COST_USD = Counter("llm_cost_usd_total", "Estimated accumulated cost in USD for LLM usage", ["provider"], registry=_METRICS_REGISTRY)
LLM_LATENCY = Histogram("llm_latency_seconds", "Histogram of end-to-end latency for LLM calls (seconds)", ["provider"], registry=_METRICS_REGISTRY, buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0))
LLM_TTFT = Histogram("llm_ttft_seconds", "Histogram of time-to-first-token (TTFT) for LLM responses (seconds)", ["provider"], registry=_METRICS_REGISTRY, buckets=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0))

def _cost_for_tokens(provider: str, total_tokens: int) -> float:
    try:
        cost_per_1k = _COST_MAP.get(provider.lower(), _COST_MAP["default"])
        return (total_tokens / 1000.0) * float(cost_per_1k)
    except Exception:
        return 0.0

def safe_record(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            with _metrics_lock:
                return func(*args, **kwargs)
        except Exception:
            return None
    return wrapped

@safe_record
def record_llm_usage(
    provider: str = "default",
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_s: Optional[float] = None,
    ttft_s: Optional[float] = None,
    cost_usd: Optional[float] = None,
) -> None:
    p = (provider or "default").lower()
    LLM_CALLS.labels(provider=p).inc(1)
    if input_tokens:
        INPUT_TOKENS.labels(provider=p).inc(int(input_tokens))
    if output_tokens:
        OUTPUT_TOKENS.labels(provider=p).inc(int(output_tokens))
    total = int((input_tokens or 0) + (output_tokens or 0))
    if total:
        TOKENS_TOTAL.labels(provider=p).inc(total)
    if latency_s is not None:
        LLM_LATENCY.labels(provider=p).observe(float(latency_s))
    if ttft_s is not None:
        LLM_TTFT.labels(provider=p).observe(float(ttft_s))
    est_cost = cost_usd if cost_usd is not None else _cost_for_tokens(p, total)
    if est_cost:
        LLM_COST_USD.labels(provider=p).inc(float(est_cost))

def record_llm_call(
    provider: str = "default",
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_s: Optional[float] = None,
    ttft_s: Optional[float] = None,
) -> None:
    record_llm_usage(
        provider=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_s=latency_s,
        ttft_s=ttft_s,
    )

def metrics_endpoint():
    try:
        data = generate_latest(_METRICS_REGISTRY)
        return data, CONTENT_TYPE_LATEST
    except Exception:
        return b"", CONTENT_TYPE_LATEST

def attach_metrics(app: "FastAPI | None") -> None:
    if not FASTAPI_AVAILABLE or app is None:
        return

    async def _fastapi_metrics():
        payload = generate_latest(_METRICS_REGISTRY)
        return Response(content=payload, media_type=CONTENT_TYPE_LATEST)

    try:
        existing = any(getattr(r, "path", "") == "/metrics" for r in getattr(app, "routes", []))
        if not existing:
            app.add_api_route("/metrics", _fastapi_metrics, methods=["GET"])
    except Exception:
        try:
            app.router.add_api_route("/metrics", _fastapi_metrics, methods=["GET"])
        except Exception:
            pass

def get_metrics_snapshot() -> dict:
    try:
        return {"metrics": "registered", "note": "use /metrics endpoint and Prometheus to view time-series"}
    except Exception:
        return {"metrics": "unavailable"}

__all__ = [
    "record_llm_call",
    "attach_metrics",
    "get_metrics_snapshot",
]
