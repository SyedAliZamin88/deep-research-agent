# Deep Research Agent – Architecture Overview

## 1. System Goals and Overview

- Perform autonomous due diligence on people or entities to uncover hidden facts and risks.
- Orchestrate multi-model AI workflows chaining planner, search, extraction, validation, and reporting.
- Provide detailed risk reports and connection graphs.
- Observe all pipeline steps and resource usage with integrated telemetry.

---

## 2. System Architecture

```
┌──────────────┐
│ CLI / FastAPI│ (scripts/run_agent.py, future integrations)
└──────┬───────┘
       │ invokes
┌──────▼───────┐
│ResearchOrch. │ (core/orchestrator.py)
└──────┬───────┘
       │ configures
┌──────▼───────┐
│ ResearchGraph│ (agents/langgraph/research_graph.py)
│ (LangGraph)  │
└──────────────┘
    │   │   │   │    │
 Planner Search Extract Validate Report

```

Each LangGraph node takes and mutates an `InvestigationState` accumulating findings, leads, context, and logs.

---

## 3. Core Components

### 3.1 Configuration
- Loads secrets, API keys, and runtime options from `.env` (config/settings.py).

### 3.2 Core State and Orchestration
- `InvestigationState`: holds subject, objectives, findings, risks, connections, logs.
- `ResearchOrchestrator` orchestrates LangGraph workflow and normalizes state.

### 3.3 LangGraph Workflow
- Nodes:
  - Planner: Generates investigation plan & initial queries.
  - Search: Executes web/API searches, collects artifacts.
  - Extraction: Extracts facts, risks, leads from raw data.
  - Validation: Runs analytics for fact validation, risk scoring, resolving entities.
  - Reporting: Synthesizes narrative risk report and slide outlines.

- Supports iterative refinement when new leads identified.

### 3.4 Provider Abstractions
- LLM providers: OpenAI, Gemini, Ollama with unified async interface.
- Search providers: Tavily, SerpApi, Web scraper.

### 3.5 Analytics Layer
- Fact validation, risk scoring, entity resolution, connection graphing feed into validation node for decision making.

### 3.6 Reporting
- Jinja2 templates produce Markdown reports and presentation outlines.

---

## 4. Data Flow Summary

1. Input: Subject and objectives via CLI or API.
2. Planning: LLM generates plan and search queries.
3. Search: Queries executed, artifacts collected.
4. Extraction: Facts, leads, risks extracted.
5. Validation: Deterministic checks, graph building, risk scoring.
6. Reporting: Narrative briefing and markdown/slides output.
7. Persistence: JSON and Markdown reports saved.
8. Logging: Aggregated logs streamed to frontend.

---

## 5. Observability and Reliability

- Logging with structlog.
- Retry and rate limiting wrappers.
- Error handling with custom exceptions.

---

## 6. Observability and Telemetry Integration

- **Tracing:**
  - OpenTelemetry + OpenInference adapter.
  - Node spans, LLM call spans with token counts and latency.
  - Jaeger trace export, UI visualizations.

- **Metrics:**
  - Prometheus metrics included for LLM calls, tokens, latency, cost.
  - Backend exposes `/metrics` for Prometheus scraping.
  - Grafana dashboards visualize telemetry data.

- **Frontend telemetry:**
  - React UI shows live token count, queries, live refinement, connection graph.

---

## 7. Deployment and Environment

- Uvicorn backend listens on all interfaces for correct metrics scraping.
- `.env` configures API keys, search provider, observability options.
- Docker Compose stack brings up Jaeger, Prometheus, Grafana, OpenTelemetry Collector (optional).

---

*Last updated: 2026-02-09*

## 8. Execution Modes

- **CLI (primary):** `python -m uv run python scripts/run_agent.py ...`
- **Local Testing:** Use Ollama + DuckDuckGo web scraper for zero-cost runs.
- **Production:** Configure OpenAI/Gemini/Tavily/SerpAPI via `.env`, adjust concurrency and timeout settings.

---

## 9. Future Enhancements

- Automated evaluation harness leveraging `data/personas` to compute precision/recall per persona.
- Live tracing/telemetry pipeline for step-level monitoring.
- Web UI for interactive review and approval of findings.
- Policy engine to enforce jurisdictional constraints (e.g., data privacy compliance).

---

*Last updated: {{ 2026-02-09 }}*
