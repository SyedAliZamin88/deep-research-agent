from __future__ import annotations
from typing import Any, Iterable, Mapping, Union
from deep_research_agent.core.state import InvestigationState
from deep_research_agent.llm_providers import ProviderType, create_provider
from deep_research_agent.reports.report_builder import RiskReportBuilder
from .base import BaseAgentNode

class ReportingNode(BaseAgentNode):
    """Produces narrative briefings and persists structured report artifacts."""

    def __init__(
        self,
        provider: ProviderType,
        *,
        report_builder: RiskReportBuilder | None = None,
    ) -> None:
        super().__init__("reporting")
        self.provider = create_provider(provider)
        self.builder = report_builder or RiskReportBuilder()

    def run(self, state: Union[InvestigationState, dict]) -> Union[InvestigationState, dict]:
        """
        Main entry point - handles both dict and InvestigationState.

        CRITICAL FIX: This method now safely handles both state types.
        """

        was_dict = isinstance(state, dict)

        if was_dict:
            state = self._dict_to_state(state)

        self.log(
            "reporting.start",
            findings=len(state.findings),
            validated=len(state.context.get("validated_facts", [])),
            risks=len(state.risks),
        )

        prompt = self._compose_prompt(state)

        def _invoke() -> str:
            return self._await_result(
                self.provider.invoke(
                    prompt,
                    system_prompt=(
                        "You are generating a due-diligence briefing. "
                        "Synthesize validated facts, risks, and connections into a concise narrative. "
                        "Highlight confidence levels, unresolved questions, and recommended next steps."
                    ),
                    temperature=0.25,
                )
            )

        summary = self.retry_policy.wrap(_invoke)()
        state.context["report_draft"] = summary
        state.record_log("reporting.summary", {"length": len(summary)})

        artifacts = self._persist_reports(state)
        if artifacts:
            state.context.setdefault("report_artifacts", {}).update(
                {name: str(path) for name, path in artifacts.items()}
            )
            state.record_log(
                "reporting.artifacts",
                {"artifacts": {name: str(path) for name, path in artifacts.items()}},
            )

        self.log(
            "reporting.complete",
            summary_length=len(summary),
            artifact_count=len(artifacts),
        )

        if was_dict:
            return self._state_to_dict(state)

        return state

    def _dict_to_state(self, data: dict) -> InvestigationState:
        """Convert dict to InvestigationState safely."""
        return InvestigationState(
            subject=data.get("subject", ""),
            objectives=data.get("objectives", []),
            findings=data.get("findings", []),
            leads=data.get("leads", []),
            risks=data.get("risks", []),
            connections=data.get("connections", []),
            context=data.get("context", {}),
            logs=data.get("logs", [])
        )

    def _state_to_dict(self, state: InvestigationState) -> dict:
        """Convert InvestigationState back to dict for LangGraph."""
        return {
            "subject": state.subject,
            "objectives": state.objectives,
            "findings": state.findings,
            "leads": state.leads,
            "risks": state.risks,
            "connections": state.connections,
            "context": state.context,
            "logs": state.logs,
        }

    def _compose_prompt(self, state: InvestigationState) -> str:
        validated_facts = self._format_validated_facts(
            state.context.get("validated_facts", [])
        )
        risk_signals = self._format_risk_signals(
            state.context.get("risk_signals", []),
            state.context.get("risk_summary"),
        )
        key_leads = self._format_items(state.leads or state.context.get("identified_leads", []), 8)
        connection_summary = self._summarize_connections(
            state.context.get("connection_graph", {})
        )

        return (
            f"Subject: {state.subject}\n"
            f"Objectives: {', '.join(state.objectives)}\n"
            f"Validated Facts:\n{validated_facts}\n"
            f"Risk Assessment:\n{risk_signals}\n"
            f"Connection Highlights:\n{connection_summary}\n"
            f"Recommended Next Leads:\n{key_leads}\n"
            "Draft a structured briefing with sections:\n"
            "1. Executive Overview\n"
            "2. Validated Findings (with confidence)\n"
            "3. Risk Outlook (severity + confidence)\n"
            "4. Relationship Insights\n"
            "5. Recommended Analyst Actions\n"
            "Keep it under 450 words."
        )

    def _format_validated_facts(self, facts: Iterable[Mapping[str, Any]]) -> str:
        lines: list[str] = []
        for index, fact in enumerate(facts):
            if index >= 6:
                break
            text = fact.get("fact") or fact.get("normalized_fact") or ""
            confidence = fact.get("confidence", "unknown")
            support = fact.get("support_mentions", fact.get("supporting_mentions", 0))
            domains = fact.get("unique_domains", 0)
            lines.append(f"- {text} (confidence: {confidence}, mentions: {support}, domains: {domains})")
        return "\n".join(lines) if lines else "- No validated facts available."

    def _format_risk_signals(
        self,
        signals: Iterable[Mapping[str, Any]],
        summary: Mapping[str, Any] | None,
    ) -> str:
        lines: list[str] = []
        if summary:
            overall = summary.get("overall_score")
            severity = summary.get("highest_severity")
            if overall is not None and severity:
                lines.append(f"*Overall score:* {overall:.2f} | *Peak severity:* {severity}")
        for index, signal in enumerate(signals):
            if index >= 6:
                break
            label = signal.get("label", "Unnamed risk")
            category = signal.get("category", "general")
            severity = signal.get("severity", "unknown")
            confidence = signal.get("confidence", "unknown")
            rationale = signal.get("rationale", "")
            lines.append(
                f"- {label} [{category}] severity={severity}, confidence={confidence}. {rationale}"
            )
        if not lines:
            lines.append("- No discrete risk signals recorded.")
        return "\n".join(lines)

    def _format_items(self, items: Iterable[str], limit: int) -> str:
        limited = [item for item in items if item][:limit]
        if not limited:
            return "- No items."
        return "\n".join(f"- {item}" for item in limited)

    def _summarize_connections(self, graph: Mapping[str, Any]) -> str:
        """
        CRITICAL FIX 4: Safe handling of connection graph metrics.
        Prevents division by zero and None formatting errors.
        """
        if not graph:
            return "- No relationship data."

        nodes = len(graph.get("nodes", []))
        edges = len(graph.get("edges", []))
        density = graph.get("density")
        centrality = graph.get("centrality", {})

        if density is not None:
            density_str = f"- Nodes: {nodes}, Edges: {edges}, Density: {density:.3f}"
        else:
            density_str = f"- Nodes: {nodes}, Edges: {edges}, Density: N/A"

        lines = [density_str]

        if centrality and isinstance(centrality, dict) and len(centrality) > 0:
            top_entities = sorted(
                centrality.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:5]

            if top_entities:
                lines.append("- Key influencers:")
                for name, score in top_entities:
                    lines.append(f"  - {name} (centrality {score:.3f})")

        return "\n".join(lines)

    def _persist_reports(self, state: InvestigationState) -> dict[str, Any]:
        """
        CRITICAL FIX 7: Safe report persistence with proper error handling.
        """
        try:
            bundle = self.builder.write_bundle(state)
            return bundle if bundle else {}
        except ZeroDivisionError as exc:
            self.log("reporting.persist_failed", error=f"Division error: {str(exc)}")

            return {}
        except Exception as exc:
            # General error handling
            self.log("reporting.persist_failed", error=str(exc))
            return {}
