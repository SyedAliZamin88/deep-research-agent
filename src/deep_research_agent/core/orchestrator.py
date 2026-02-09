from __future__ import annotations
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
import orjson
from deep_research_agent.agents.langgraph import ResearchGraph, ResearchGraphConfig
from deep_research_agent.config import settings
from deep_research_agent.core.errors import ProviderNotConfiguredError
from deep_research_agent.core.state import InvestigationState, ResearchArtifact
from deep_research_agent.utils import configure_logging, get_logger


class ResearchOrchestrator:
    """High-level entry point responsible for executing the research workflow."""

    def __init__(self, config: ResearchGraphConfig | None = None) -> None:
        configure_logging()
        self.logger = get_logger(__name__)
        self.settings = settings
        self.config = config or self._build_config_from_settings()
        self.graph = ResearchGraph(self.config)

    def run(self, subject: str, objectives: list[str]) -> InvestigationState:
        if not objectives:
            raise ValueError("At least one investigation objective must be supplied.")

        state = InvestigationState(subject=subject, objectives=objectives)
        self.logger.info(
            "orchestrator.start",
            subject=subject,
            objectives=objectives,
            graph_config=asdict(self.config),
        )

        raw_result = self.graph.run(state)

        if isinstance(raw_result, InvestigationState):
            final_state = raw_result
            self.logger.info("orchestrator.state_type", msg="Already InvestigationState")
        elif isinstance(raw_result, dict):
            self.logger.info("orchestrator.state_type", msg="Converting dict to InvestigationState")
            final_state = InvestigationState(
                subject=raw_result.get('subject', subject),
                objectives=raw_result.get('objectives', objectives),
                findings=raw_result.get('findings', []),
                risks=raw_result.get('risks', []),
                connections=raw_result.get('connections', []),
                context=raw_result.get('context', {})
            )
        else:
            self.logger.error("orchestrator.unexpected_return_type", type=type(raw_result).__name__)
            final_state = state

        self._persist_state(final_state)

        self.logger.info("orchestrator.final_state_type", type=type(final_state).__name__, is_investigation_state=isinstance(final_state, InvestigationState))

        self.logger.info(
            "orchestrator.finish",
            findings=len(final_state.findings),
            risks=len(final_state.risks),
            leads=len(final_state.context.get("identified_leads", [])),
        )

        if not isinstance(final_state, InvestigationState):
            self.logger.error("orchestrator.return_type_mismatch", type=type(final_state).__name__)
            raise TypeError(f"Expected InvestigationState, got {type(final_state)}")

        return final_state

    def _build_config_from_settings(self) -> ResearchGraphConfig:
        provider = self._select_llm_provider()
        search_provider = self._select_search_provider()

        return ResearchGraphConfig(
            planner_provider=provider,
            extraction_provider=provider,
            reporting_provider=provider,
            search_provider=search_provider,
            search_results_per_query=self.settings.runtime.max_concurrent_requests * 2,
            max_iterations=2,
        )

    def _select_llm_provider(self) -> str:
        if self.settings.secrets.openai_api_key:
            return "openai"
        if self.settings.secrets.google_api_key:
            return "gemini"
        if self.settings.secrets.llama_model_path:
            return "ollama"
        raise ProviderNotConfiguredError("No LLM provider is configured. Check .env values.")


    def _select_search_provider(self) -> str:

        default = self.settings.runtime.default_search_engine

        mapping = {"tavily": "tavily", "serpapi": "serpapi", "google": "serpapi"}

        provider = mapping.get(default, "tavily")



        if provider == "tavily" and not self.settings.secrets.tavily_api_key:

            self.logger.warning(
                "orchestrator.provider_fallback",
                requested=default,
                missing_secret="TAVILY_API_KEY",
                fallback="serpapi",

            )
            provider = "serpapi"

        if provider == "serpapi" and not self.settings.secrets.serpapi_api_key:

            self.logger.warning(
                "orchestrator.provider_fallback",
                requested=default,
                missing_secret="SERPAPI_API_KEY",
                fallback="web",

            )
            provider = "web"

        if provider == "web":
            self.logger.warning(
                "orchestrator.provider_fallback",
                requested=default,
                missing_secret="ALL_SEARCH_API_KEYS",
                fallback="web",
                message="Using web scraper fallback. Search coverage may be limited.",
            )


        return provider

    def _coerce_state(self, raw: Any, fallback: InvestigationState) -> InvestigationState:
        """
        Convert LangGraph dict output back to InvestigationState.
        LangGraph returns dicts, but we need InvestigationState objects.
        """
        if isinstance(raw, InvestigationState):
            return raw

        if not isinstance(raw, dict):
            self.logger.warning("orchestrator.unexpected_type", type=type(raw).__name__)
            return fallback

        try:
            # Create new InvestigationState from dict
            return InvestigationState(
                subject=raw.get('subject', fallback.subject),
                objectives=raw.get('objectives', fallback.objectives),
                findings=raw.get('findings', []),
                risks=raw.get('risks', []),
                connections=raw.get('connections', []),
                context=raw.get('context', {})
            )
        except Exception as e:
            self.logger.error("orchestrator.coercion_error", error=str(e))
            return fallback

    @staticmethod
    def _normalize_payload(data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise TypeError(f"Expected mapping payload from LangGraph, received {type(data)!r}")

        candidate = data
        visited: set[int] = set()
        while isinstance(candidate, dict):
            marker = id(candidate)
            if marker in visited:
                break
            visited.add(marker)

            next_candidate = None
            for key in ("state", "value"):
                nested = candidate.get(key)
                if isinstance(nested, dict):
                    next_candidate = nested
                    break

            if next_candidate is None:
                break
            candidate = next_candidate

        if not isinstance(candidate, dict):
            raise TypeError(f"Unable to interpret graph state payload of type {type(candidate)!r}")

        return candidate

    def _coerce_finding(self, artifact: Any) -> ResearchArtifact:
        if isinstance(artifact, ResearchArtifact):
            return artifact
        if isinstance(artifact, dict):
            return ResearchArtifact(
                title=artifact.get("title") or "untitled",
                url=artifact.get("url") or "",
                snippet=artifact.get("snippet") or "",
                timestamp=artifact.get("timestamp"),
                confidence=artifact.get("confidence"),
                metadata=artifact.get("metadata") or {},
            )
        raise TypeError(f"Unsupported finding payload type: {type(artifact)!r}")



    def _persist_state(self, state: InvestigationState | dict[str, Any]) -> None:
        """Save state artifacts and logs to disk."""
        try:
            if isinstance(state, dict):
                subject = state.get('subject', 'unknown')
                context = state.get('context', {})
            else:
                subject = state.subject
                context = state.context

            logs = context.get('logs', [])
            if logs:
                timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
                log_path = self.config.reports_dir / f"{timestamp}_{subject.lower().replace(' ', '_')}_execution.json"
                log_path.parent.mkdir(parents=True, exist_ok=True)

                with open(log_path, 'w', encoding='utf-8') as f:
                    json.dump({'logs': logs}, f, indent=2, default=str)

                self.logger.info("orchestrator.logs_saved", path=str(log_path), log_count=len(logs))

            if hasattr(state, 'record_log'):
                state.record_log("orchestrator.persisted", {"subject": subject})
            elif isinstance(state, dict) and 'logs' in state:
                state['logs'].append({
                    "event": "orchestrator.persisted",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "payload": {"subject": subject}
                })

        except Exception as e:
            self.logger.error("orchestrator.persist_error", error=str(e))

    def _state_to_payload(self, state: InvestigationState) -> dict[str, Any]:
        findings_payload = [
            {
                "title": artifact.title,
                "url": artifact.url,
                "snippet": artifact.snippet,
                "timestamp": artifact.timestamp,
                "confidence": artifact.confidence,
                "metadata": artifact.metadata,
            }
            for artifact in state.findings
        ]

        return {
            "subject": state.subject,
            "objectives": state.objectives,
            "findings": findings_payload,
            "risks": state.risks,
            "leads": state.leads,
            "connections": state.connections,
            "context": state.context,
            "logs": state.logs,
        }

    @staticmethod
    def _slugify(value: str) -> str:
        cleaned = "".join(ch if ch.isalnum() else "_" for ch in value.lower())
        normalized = "_".join(filter(None, cleaned.split("_")))
        return normalized or "subject"
