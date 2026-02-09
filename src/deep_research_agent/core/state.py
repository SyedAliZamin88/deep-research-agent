from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ResearchArtifact:
    """Individual search result or finding."""
    title: str
    url: str
    snippet: str
    timestamp: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class InvestigationState:
    """Main state object for the research workflow."""
    subject: str
    objectives: list[str]
    findings: list[ResearchArtifact] = field(default_factory=list)
    leads: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    connections: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)

    def add_finding(self, artifact: ResearchArtifact) -> None:
        self.findings.append(artifact)

    def add_lead(self, lead: str) -> None:
        if lead not in self.leads:
            self.leads.append(lead)

    def add_risk(self, risk: str) -> None:
        self.risks.append(risk)

    def add_connection(self, connection: dict[str, Any]) -> None:
        self.connections.append(connection)

    def record_log(self, event: str, payload: dict[str, Any]) -> None:
        self.logs.append({"event": event, "payload": payload})
