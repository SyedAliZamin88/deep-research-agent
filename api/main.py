from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio
import uuid

from deep_research_agent.core.orchestrator import ResearchOrchestrator
from deep_research_agent.core.state import InvestigationState

from fastapi.staticfiles import StaticFiles
from deep_research_agent.observability.prometheus_metrics import attach_metrics, get_metrics_snapshot

# OpenTelemetry minimal setup with Jaeger exporter and console exporter
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter

    _resource = Resource.create({"service.name": "deep-research-agent"})
    _provider = TracerProvider(resource=_resource)
    trace.set_tracer_provider(_provider)

    _provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    _jaeger_exporter = JaegerExporter(
        collector_endpoint="http://localhost:14268/api/traces",
    )
    _provider.add_span_processor(BatchSpanProcessor(_jaeger_exporter))

    _otel_tracer = trace.get_tracer(__name__)
    _otel_available = True
except Exception:
    _otel_tracer = None
    _otel_available = False

app = FastAPI(title="Deep Research AI Agent API")

# Attach Prometheus /metrics endpoint to FastAPI app
attach_metrics(app)

# Optional OpenTelemetry middleware to trace each HTTP request
if _otel_available:
    try:
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request

        class OpenTelemetryRequestMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                span_name = f"HTTP {request.method} {request.url.path}"
                with _otel_tracer.start_as_current_span(span_name):
                    response = await call_next(request)
                return response

        app.add_middleware(OpenTelemetryRequestMiddleware)
    except Exception:
        pass

# CORS middleware for local frontend development
allow_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve investigation reports statically
app.mount("/data/reports", StaticFiles(directory="data/reports"), name="reports")

# In-memory job tracking
jobs_status: Dict[str, str] = {}
jobs_logs: Dict[str, List[str]] = {}
jobs_results: Dict[str, Any] = {}

class InvestigateRequest(BaseModel):
    subject: str
    objectives: List[str]
    llms: List[str]
    search_provider: str

def append_log(job_id: str, message: str):
    if job_id in jobs_logs:
        jobs_logs[job_id].append(message)

@app.post("/api/investigate")
async def start_investigation(req: InvestigateRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs_status[job_id] = "Queued"
    jobs_logs[job_id] = []
    jobs_results[job_id] = None

    background_tasks.add_task(run_investigation_job, job_id, req)

    return {"job_id": job_id}

async def run_investigation_job(job_id: str, req: InvestigateRequest):
    jobs_status[job_id] = "Started"
    append_log(job_id, f"Investigation started for subject: {req.subject}")
    append_log(job_id, f"Objectives: {', '.join(req.objectives)}")
    append_log(job_id, f"Using LLM providers: {', '.join(req.llms)} and search provider: {req.search_provider}")

    orchestrator = ResearchOrchestrator()

    try:
        state = await asyncio.to_thread(
            orchestrator.run,
            req.subject,
            req.objectives,
        )
    except Exception as exc:
        append_log(job_id, f"Investigation error: {str(exc)}")
        jobs_status[job_id] = "Error"
        jobs_results[job_id] = {"error": str(exc)}
        return

    findings_count = len(getattr(state, "findings", []))
    leads_count = len(getattr(state, "leads", []))
    risks_count = len(getattr(state, "risks", []))
    append_log(job_id, f"Investigation finished with {findings_count} findings, {leads_count} leads, {risks_count} risks")

    facts = state.context.get("validated_facts") or state.context.get("extracted_facts") or []
    if not facts:
        fallback_facts = []
        for f in getattr(state, "findings", [])[:10]:
            title = getattr(f, "title", "") or ""
            snippet = getattr(f, "snippet", "") or ""
            text = title.strip() if title.strip() else snippet.strip()[:300]
            if text:
                fallback_facts.append({"fact": text, "confidence": None, "support_mentions": 0})
        facts = fallback_facts

    risks = state.context.get("risk_signals") or state.context.get("identified_risks") or state.risks or []
    if not risks:
        risk_keywords = [
            "sanction",
            "fraud",
            "aml",
            "money launder",
            "fraudulent",
            "sanctioned",
            "investigation",
            "risk",
            "regulatory",
            "lawsuit",
            "sanctions",
            "penalty",
        ]
        found_risks = []
        for f in getattr(state, "findings", [])[:50]:
            text = " ".join(filter(None, [getattr(f, "title", ""), getattr(f, "snippet", "")])).lower()
            if any(kw in text for kw in risk_keywords):
                found_risks.append(
                    {"label": text[:200], "category": "heuristic", "severity": "unknown", "confidence": "low"}
                )
        risks = found_risks

    raw_connections = state.context.get("connection_graph", {}).get("edges", []) or state.connections or []

    connections = []
    for conn in raw_connections:
        source = conn.get("source") or conn.get("from") or "Unknown"
        target = conn.get("target") or conn.get("to") or "Unknown"
        relation = conn.get("relation") or conn.get("label") or "associated_with"
        connections.append(
            {
                "source": source,
                "target": target,
                "relation": relation,
            }
        )

    from pathlib import Path
    import re

    json_report_url = ""
    md_report_url = ""

    report_slug = req.subject.lower().replace(" ", "_")
    slug_tokens = [t for t in report_slug.split("_") if t]

    report_dir = Path("data/reports")

    report_artifacts = state.context.get("report_artifacts") or {}
    if report_artifacts:
        try:
            json_path = report_artifacts.get("json")
            md_path = report_artifacts.get("markdown")
            if json_path:
                json_name = Path(str(json_path)).name
                json_report_url = f"/data/reports/{json_name}"
            if md_path:
                md_name = Path(str(md_path)).name
                md_report_url = f"/data/reports/{md_name}"
        except Exception:
            json_report_url = ""
            md_report_url = ""

    def _fuzzy_candidates(ext: str):
        if not report_dir.exists() or not report_dir.is_dir():
            return []
        candidates = []
        for p in report_dir.iterdir():
            if not p.name.lower().endswith(ext):
                continue
            name_l = p.name.lower()
            if report_slug in name_l:
                candidates.append(p)
                continue
            match_count = sum(1 for tok in slug_tokens if tok and tok in name_l)
            if slug_tokens and match_count >= max(1, len(slug_tokens) // 2):
                candidates.append(p)
                continue
            if re.search(r"\d{6,}_", p.name):
                if any(tok in name_l for tok in slug_tokens):
                    candidates.append(p)
        return candidates

    if not json_report_url:
        json_candidates = _fuzzy_candidates(".json")
        if json_candidates:
            latest_json = max(json_candidates, key=lambda p: p.stat().st_mtime)
            json_report_url = f"/data/reports/{latest_json.name}"

    if not md_report_url:
        md_candidates = _fuzzy_candidates(".md")
        if md_candidates:
            latest_md = max(md_candidates, key=lambda p: p.stat().st_mtime)
            md_report_url = f"/data/reports/{latest_md.name}"

    if not json_report_url:
        json_report_url = ""
    if not md_report_url:
        md_report_url = ""

    refined_queries = []
    try:
        refined_queries = state.context.get("initial_queries") or state.context.get("refined_queries") or []
    except Exception:
        refined_queries = []

    try:
        state_logs = getattr(state, "logs", None) or state.context.get("logs", []) or []
        for entry in state_logs:
            append_log(job_id, entry if isinstance(entry, str) else str(entry))
    except Exception:
        pass

    jobs_results[job_id] = {
        "facts": facts,
        "risks": risks,
        "connections": connections,
        "refined_queries": refined_queries,
        "reports": {"json": json_report_url, "markdown": md_report_url},
    }
    jobs_status[job_id] = "Completed"


@app.websocket("/ws/status/{job_id}")
async def websocket_status(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        while True:
            if job_id not in jobs_status:
                await websocket.send_json({"error": "invalid job ID"})
                break

            status = jobs_status[job_id]
            logs = jobs_logs.get(job_id, [])
            results = jobs_results.get(job_id)

            await websocket.send_json({"status": status, "logs": logs, "results": results})

            if status == "Completed":
                break

            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
