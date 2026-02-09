from __future__ import annotations
from collections import Counter, defaultdict
from typing import Iterable
from deep_research_agent.analysis.connection_graph import build_connection_graph
from deep_research_agent.analysis.entity_resolution import (
    EntityCandidate,
    build_entity_index,
    resolve_entities,
)
from deep_research_agent.analysis.fact_validator import ValidationResult, validate_facts
from deep_research_agent.analysis.risk_scoring import (
    aggregate_risk_scores,
    build_risk_signals,
)
from deep_research_agent.core.state import InvestigationState, ResearchArtifact
from deep_research_agent.utils import get_logger
from .base import BaseAgentNode


class ValidationNode(BaseAgentNode):
    """Cross-validates extracted intelligence and enriches the investigation context."""

    def __init__(self) -> None:
        super().__init__("validation")
        self.logger = get_logger(__name__)

    def run(self, state: InvestigationState) -> InvestigationState:
        candidate_facts = list(state.context.get("extracted_facts", []))
        self.log(
            "validation.start",
            findings=len(state.findings),
            candidate_facts=len(candidate_facts),
            risks=len(state.risks),
            connections=len(state.connections),
        )

        validated_facts = validate_facts(state.findings, candidate_facts)
        state.context["validated_facts"] = [result.to_dict() for result in validated_facts]
        self._log_validated_facts(state, validated_facts)

        try:
            risk_signals = build_risk_signals(state.risks, validated_facts)
            risk_summary = aggregate_risk_scores(risk_signals)
            state.context["risk_signals"] = [signal.to_dict() for signal in risk_signals]
            state.context["risk_summary"] = risk_summary.to_dict()
            state.record_log(
                "validation.risk_summary",
                {
                    "overall_score": risk_summary.overall_score,
                    "highest_severity": risk_summary.highest_severity,
                    "signal_count": len(risk_signals),
                },
            )
        except Exception as exc:
            self.log("validation.risk_scoring_error", error=str(exc))
            state.record_log("validation.risk_scoring_error", {"error": str(exc)})
            state.context["risk_signals"] = []
            state.context["risk_summary"] = {}

        try:

            self.log("validation.build_connection_graph.start", raw_connection_count=len(state.connections))
            state.record_log("validation.build_connection_graph.start", {"raw_connection_count": len(state.connections)})
            connection_summary = build_connection_graph(state.connections).to_dict()
            state.context["connection_graph"] = connection_summary
            nodes_count = len(connection_summary.get("nodes", []))
            edges_count = len(connection_summary.get("edges", []))
            density = connection_summary.get("density")
            self.log(
                "validation.build_connection_graph.complete",
                nodes=nodes_count,
                edges=edges_count,
                density=density,
            )
            state.record_log("validation.connection_graph_summary", {"nodes": nodes_count, "edges": edges_count, "density": density})
            if edges_count == 0 and state.findings:
                self.log("validation.connection_graph.fallback_attempt", finding_count=len(state.findings))
                state.record_log("validation.connection_graph.fallback_attempt", {"finding_count": len(state.findings)})
                fallback_edges = []
                for art in state.findings[:50]:
                    title = getattr(art, "title", "") or ""
                    snippet = getattr(art, "snippet", "") or ""
                    text = f"{title} {snippet}"
                    tokens = [t.strip(".,()[]{}:;\"'") for t in text.split() if t and t[0].isupper()]
                    words = []
                    for t in tokens:
                        w = t.strip()
                        if len(w) > 1 and w not in words:
                            words.append(w)
                    if len(words) >= 2:
                        src = words[0]
                        tgt = words[1]
                        if src != tgt:
                            fallback_edges.append({"source": src, "target": tgt, "relation": "co_mentioned"})
                if fallback_edges:
                    existing_edges = connection_summary.get("edges", [])
                    for e in fallback_edges:
                        existing_edges.append({"source": e["source"], "target": e["target"], "relation": e["relation"], "weight": 1.0, "metadata": {"heuristic": True}})
                    state.context["connection_graph"]["edges"] = existing_edges
                    state.record_log("validation.connection_graph.fallback_populated", {"added_edges": len(fallback_edges)})
                    self.log("validation.connection_graph.fallback_populated", added=len(fallback_edges))

        except Exception as exc:
            self.log("validation.connection_graph_error", error=str(exc))
            state.record_log("validation.connection_graph_error", {"error": str(exc)})
            state.context["connection_graph"] = {}

        try:
            graph_nodes = state.context.get("connection_graph", {}).get("nodes", [])
            resolved_entities = self._resolve_entities(state, graph_nodes)
            state.context["resolved_entities"] = [candidate.to_dict() for candidate in resolved_entities]
            state.context["entity_index"] = {
                alias: candidate.canonical_name
                for alias, candidate in build_entity_index(resolved_entities).items()
            }

            resolved_count = len(resolved_entities)
            top_entities = [c.canonical_name for c in resolved_entities[:10]]
            self.log("validation.entities_resolved", count=resolved_count, top=top_entities)
            state.record_log("validation.entities_resolved", {"count": resolved_count, "top": top_entities})

        except Exception as exc:
            self.log("validation.entity_resolution_error", error=str(exc))
            state.record_log("validation.entity_resolution_error", {"error": str(exc)})
            state.context["resolved_entities"] = []
            state.context["entity_index"] = {}

        domain_counts = self._compute_domain_counts(state.findings)
        state.context["source_domain_counts"] = dict(domain_counts)
        state.context["source_quality"] = self._summarize_source_quality(domain_counts)

        self.log(
            "validation.complete",
            validated=len(validated_facts),
            risk_signals=len(state.context.get("risk_signals", [])),
            entities=len(state.context.get("resolved_entities", [])),
            domains=len(domain_counts),
        )
        return state

    def _log_validated_facts(
        self,
        state: InvestigationState,
        results: Iterable[ValidationResult],
        *,
        max_logs: int = 10,
    ) -> None:
        for index, result in enumerate(results):
            if index >= max_logs:
                break
            state.record_log(
                "validation.fact",
                {
                    "fact": result.fact,
                    "confidence": result.confidence,
                    "support_mentions": result.support_mentions,
                    "unique_domains": result.unique_domains,
                },
            )

    def _resolve_entities(
        self,
        state: InvestigationState,
        graph_nodes: list[dict[str, object]],
    ) -> list[EntityCandidate]:
        mentions = set(state.leads)
        mentions.update(state.context.get("identified_leads", []))
        for node in graph_nodes:
            name = node.get("name")
            if isinstance(name, str):
                mentions.add(name)

        source_index: dict[str, list[str]] = defaultdict(list)
        for connection in state.connections:
            source = str(connection.get("source") or connection.get("from") or "").strip()
            target = str(connection.get("target") or connection.get("to") or "").strip()
            evidence = connection.get("evidence")
            if source:
                source_index[source].append(str(evidence))
            if target:
                source_index[target].append(str(evidence))

        resolved = resolve_entities(
            mentions,
            source_index={k: [v for v in values if v] for k, values in source_index.items()},
        )
        self.log("validation.entities_resolved", count=len(resolved))
        return resolved

    def _compute_domain_counts(self, findings: Iterable[ResearchArtifact]) -> Counter[str]:
        counts: Counter[str] = Counter()
        for artifact in findings:
            domain = ""
            if artifact.metadata and isinstance(artifact.metadata, dict):
                domain = str(artifact.metadata.get("domain") or "").lower()
            if not domain and artifact.url:
                parts = artifact.url.split("/")
                if len(parts) >= 3:
                    domain = parts[2].lower()
            if domain:
                counts[domain] += 1
        return counts

    def _summarize_source_quality(self, domain_counts: Counter[str]) -> dict[str, object]:
        if not domain_counts:
            return {"unique_domains": 0, "top_domains": [], "diversity_score": 0.0}

        total_mentions = sum(domain_counts.values())
        unique_domains = len(domain_counts)
        diversity_score = round(unique_domains / total_mentions, 3) if total_mentions else 0.0
        top_domains = domain_counts.most_common(5)

        return {
            "unique_domains": unique_domains,
            "top_domains": top_domains,
            "diversity_score": diversity_score,
        }
