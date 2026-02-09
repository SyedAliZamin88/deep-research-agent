# Deep Research AI Agent

## Overview
Deep Research AI Agent is an autonomous research system capable of conducting thorough investigations on individuals or entities to uncover hidden connections, potential risks, and strategic insights. It integrates multiple AI models, supports dynamic search query refinement, and includes a robust observability stack to monitor telemetry and tracing.

## Features
- **Multi-model AI orchestration:** Supports OpenAI, Google Gemini, and Ollama LLaMA models.
- **Dynamic Query Refinement:** Agent adapts search strategies based on newly discovered leads.
- **Deep Fact Extraction & Risk Recognition**
- **Connection Mapping & Source Validation**
- **Observability Stack:** Integrated metrics and tracing via Prometheus, Jaeger, and Grafana.
- **Evaluation Harness:** Test personas for assessing AI research quality.

---

## Prerequisites
- Python 3.12+
- Node.js 18+
- Docker Desktop (for running observability stack)
- API keys for preferred AI models (OpenAI, Google Gemini, Tavily, etc.)

---

## Getting Started

### 1. Clone the Repository
```bash
git clone <your_repo_url>
cd deep-research-agent
```

### 2. Setup Python Environment and Install Dependencies
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Unix/Mac:
source .venv/bin/activate

pip install -U pip
pip install -e ".[observability,dev]"
```

### 3. Setup Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

### 4. Configure Environment Variables
- Copy `.env.example` to `.env`:
```bash
copy .env.example .env     # Windows PowerShell
# or
cp .env.example .env       # Unix/Mac
```
- Edit `.env` and provide your API keys (Get keys from your platform):

```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_gemini_key_here
TAVILY_API_KEY=your_tavily_key_here
SERPAPI_API_KEY=your_serpapi_key_here
LLAMA_MODEL_PATH=            # if using LLaMA locally
DEFAULT_SEARCH_ENGINE=tavily
OTEL_SERVICE_NAME=deep-research-agent
COST_PER_1K_OPENAI=0.03
COST_PER_1K_GEMINI=0.03
```

### 5. Start Backend API Server
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Start Frontend Dev Server (In a new terminal)
```bash
cd frontend
npm run dev
```
- Access UI at: http://localhost:5173

### 7. Start Observability Stack with Docker Compose
```bash
docker compose up -d
```
This brings up:
- **Jaeger:** Tracing UI at http://localhost:16686
- **Prometheus:** Metrics UI at http://localhost:9090
- **Grafana:** Dashboard UI at http://localhost:3000  (default admin/admin or admin1/admin1)

Use `docker ps` to verify containers are running.

### 8. Run Investigations in UI
- Provide subject and objectives (default sample persona `Aleena Farrow` provided)
- Select at least two LLM models and a search provider.
- Click **Run Investigation**
- Watch live Activity Logs, Queries, Results, Connection Graph, and Telemetry panels.

### 9. View Metrics & Traces
- **Jaeger UI:** View request traces, node spans, and LLM call timings.
- **Prometheus UI:** Query metrics such as `llm_calls_total`, `llm_tokens_total`, `llm_cost_usd_total`.
- **Grafana UI:** View dashboards visualizing metrics and costs live.

### 10. Run Evaluation Harness
```bash
python scripts/evaluate_personas.py
```
- Automatically runs investigations on predefined personas.
- Generates detailed JSON reports and scoring summaries.

---

## Directory Structure

| Directory/File                | Description                                       |
|------------------------------|-------------------------------------------------|
| `api/`                       | Backend API and server                            |
| `data/personas/`             | Test personas with hidden facts                   |
| `data/reports/`              | Generated investigation reports                   |
| `frontend/`                  | React frontend UI                                |
| `scripts/`                   | Utility scripts (evaluation, metrics emit)       |
| `src/deep_research_agent/`   | Core agent code, AI providers, observability     |
| `docker-compose.yml`         | Observability stack compose file                  |
| `prometheus.yml`             | Prometheus scrape config                          |
| `otel-collector-config.yaml` | OpenTelemetry Collector config (optional)        |
| `.env.example`               | Environment variable template                     |
| `README.md`                  | This README                                       |

---

## Notes & Tips

- Make sure API keys in `.env` are valid for best results.
- The backend binds to all interfaces at host 0.0.0.0 for Prometheus scraping.
- Use the `scripts/emit_metric.py` to generate test metrics manually.
- Grafana dashboards can be created via UI or import JSON (ask for dashboard JSON if needed).
- For fast demos, use the UI and observe live telemetry panels and Jaeger traces.
- The system supports fallback search providers for zero-cost testing.

---

## Troubleshooting

- Backend startup errors: ensure Python packages installed (`pip install -e ".[observability,dev]"`).
- Prometheus not scraping backend: check `prometheus.yml` targets and backend bind IP (`0.0.0.0`).
- Docker networking: ensure Prometheus can reach backend IP; adjust `prometheus.yml` target accordingly.
- Container conflicts: use `docker ps` and `docker rm -f <container>` to clear stale containers.
- Telemetry missing: run investigations or `scripts/emit_metric.py` to inject fake load.
- UI errors: check browser console and backend logs for details.

---

## Contact & Support

For questions or help preparing your demo, please reach out at your convenience.

---

*Last updated: `YYYY-MM-DD`*
