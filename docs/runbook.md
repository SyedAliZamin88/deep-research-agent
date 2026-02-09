# Deep Research Agent – Operational Runbook

## 1. Purpose
This runbook provides step-by-step procedures for operating, monitoring, and troubleshooting the Deep Research Agent. It is intended for engineers and analysts responsible for running assessments, maintaining provider integrations, and validating outputs.

---

## 2. Prerequisites

### 2.1 System Requirements
- OS: Windows 10/11 (primary target) or Linux/macOS for server deployments.
- Python: 3.12 (managed via `uv`).
- Disk space: ≥ 5 GB free (report artifacts, search caches).
- Network: HTTPS access to selected LLM/search providers.

### 2.2 Install Dependencies
1. Install [uv](https://github.com/astral-sh/uv) (Python project manager).
2. Clone repository and create virtual environment:
   ```
   git clone <repo-url>
   cd deep-research-agent
   python -m uv sync
   ```
3. Verify installation:
   ```
   python -m uv run python - <<'PY'
   import deep_research_agent
   print("Deep Research Agent version:", deep_research_agent.__version__)
   PY
   ```

### 2.3 Configure Environment Variables
Copy `.env.example` to `.env` and set values:

| Variable | Description | Required for |
|----------|-------------|---------------|
| `OPENAI_API_KEY` | OpenAI API key | OpenAI LLM usage |
| `GOOGLE_API_KEY`, `GOOGLE_CSE_ID` | Google Custom Search | Google search provider |
| `SERPAPI_API_KEY` | SerpAPI key | SerpAPI provider |
| `TAVILY_API_KEY` | Tavily key | Tavily provider |
| `LLAMA_MODEL_PATH` | Directory containing local Ollama models | Local LLM fallback |
| `REQUEST_TIMEOUT`, `MAX_CONCURRENT_REQUESTS`, `LOG_LEVEL` | Runtime controls | General |

For low-cost/local testing:
```
DEFAULT_SEARCH_ENGINE=web
LLAMA_MODEL_PATH=C:\Users\<user>\.ollama\models
```

---

## 3. Execution Procedures

### 3.1 One-Off Investigation (CLI)
```
python -m uv run python scripts/run_agent.py "Subject Name" ^
    --objective "Primary objective" ^
    --objective "Secondary objective" ^
    --planner-provider ollama ^
    --extraction-provider ollama ^
    --reporting-provider ollama ^
    --search-provider web
```

**Recommended flags:**
- `--search-results-per-query <int>` – controls fetch volume (default: `2 * MAX_CONCURRENT_REQUESTS`).
- `--max-iterations <int>` – number of LangGraph refinement loops (default: 2).
- `--objectives-file path.txt` – one objective per line.

### 3.2 Batch Runs
For multiple subjects, prepare a CSV with columns `subject`, `objective1`, `objective2`, …; iterate via PowerShell or Python script that invokes the CLI.

### 3.3 Report Artifacts
- JSON state dumps: `data/reports/<timestamp>_<slug>.json`
- Markdown risk report: `<timestamp>_<slug>_risk_report.md`
- Slide outline: `<timestamp>_<slug>_briefing_outline.md`
- Execution logs: `data/logs/` (if enabled)
- Persona evaluation outputs (optional): `tests/results/` (future enhancement)

---

## 4. Provider Configuration Matrix

| Scenario | Planner | Extraction | Reporting | Search | Notes |
|----------|---------|-----------|-----------|--------|-------|
| Low-cost local | Ollama (`llama3`) | Ollama | Ollama | Web (DuckDuckGo) | Requires `ollama` service running |
| Mixed mode | Gemini | OpenAI | OpenAI | Tavily | Highest accuracy; ensure rate limits |
| Fallback | OpenAI | OpenAI | OpenAI | SerpAPI | Requires `SERPAPI_API_KEY` |
| No API keys | Ollama | Ollama | Ollama | Web | Minimal coverage; slower refinement |

**Switch providers via CLI flags or adjust `ResearchGraphConfig` instantiation.**

---

## 5. Monitoring & Logging

### 5.1 Runtime Logs
- Structured logging via `structlog`; default INFO level.
- Logs include node-level events (`agent.planner`, `agent.search`, etc.) and orchestrator lifecycle.
- Enable DEBUG for troubleshooting:
  ```
  LOG_LEVEL=DEBUG
  ```

### 5.2 Metrics & Tracing (Optional)
- Future integration: export to OpenTelemetry (placeholders in config).
- For now, rely on structured logs and context entries (`state.logs`).

---

## 6. Validation & QA

### 6.1 Built-In Analytics
`ValidationNode` stores:
- `validated_facts` with evidence and confidence.
- `risk_summary` (overall score, category breakdown).
- `risk_signals` with severity/confidence.
- `connection_graph` summary (centrality, density).
- `resolved_entities` alias clusters.
- `source_quality` (domain diversity).

Review these in the JSON state or Markdown report before sharing externally.

### 6.2 Evaluation Personas (Baseline Testing)
Files under `data/personas/`:
- `persona_alpha.json`
- `persona_beta.json`
- `persona_gamma.json`

Each includes hidden facts, expected findings, red herrings, scoring guidance. Use them to validate new model/provider configurations by checking whether the agent surfaces the “must_surface” items.

### 6.3 Automated Tests
Run unit and integration tests:
```
python -m uv run pytest
```
Add mocks for providers to keep tests offline. Review coverage report and address failures before deployment.

---

## 7. Troubleshooting Guide

| Symptom | Likely Cause | Resolution |
|---------|--------------|-----------|
| `ProviderNotConfiguredError` | Missing API key or `LLAMA_MODEL_PATH` unset | Update `.env`, restart session |
| Search returns zero results | Provider quota reached or network issue | Switch search provider flag, verify credentials |
| Reports missing | File permissions or disk full | Check `data/reports` permissions, free disk space |
| Stuck on iterative refinement | Too many leads triggering loops | Lower `--max-iterations` or refine extraction prompts |
| LLM call timeouts | Provider latency | Increase `REQUEST_TIMEOUT`, reduce concurrency |
| Markdown report lacks data | Validation context empty | Ensure `extraction` node produced facts; verify `analysis/` modules |

---

## 8. Change Management

1. Create feature branch, implement changes.
2. Update tests and documentation (architecture, runbook, personas if needed).
3. Run lint/tests; capture results.
4. Submit PR with summary, validation evidence, and sample run logs.
5. Merge after review and re-run baseline personas.

---

## 9. Security & Compliance

- Store API keys in `.env` only; do not hardcode in codebase.
- Reports may contain sensitive information; enforce access controls on `data/`.
- Respect provider ToS and jurisdictional constraints when handling personal data.
- For production deployments, integrate with secrets manager and centralized logging (future roadmap).

---

## 10. Operational Checklist

### Before Run
- [ ] `.env` populated with required provider credentials.
- [ ] `ollama` service running (if using local models).
- [ ] Network connectivity verified.
- [ ] Objective list validated (no typos, scoped queries).

### During Run
- [ ] Monitor CLI output for errors or warnings.
- [ ] Observe log entries for node progress.
- [ ] If using paid APIs, monitor usage dashboards.

### After Run
- [ ] Review Markdown report for coherence and accuracy.
- [ ] Inspect JSON state (validated facts, risk signals).
- [ ] Archive artifacts according to data retention policy.
- [ ] Log issues or anomalies in ticketing system.

---

## 11. Contact & Escalation

- Primary engineering contact: `<name/email>`
- AI/ML support channel: `#deep-research-agent` (internal chat)
- Emergency escalation: notify platform on-call engineer and disable API keys if abnormal usage detected.

---

*End of Runbook*