from __future__ import annotations
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
import orjson
from jinja2 import Environment, FileSystemLoader, select_autoescape
from deep_research_agent.config import settings
from deep_research_agent.core.state import InvestigationState
from deep_research_agent.utils import get_logger

class RiskReportBuilder:
    """Generates structured risk assessment reports from investigation state."""

    def __init__(self, template_dir: Path | None = None) -> None:
        self.logger = get_logger(__name__)
        self.template_dir = template_dir or self._default_template_dir()

        if self.template_dir.exists():
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                autoescape=select_autoescape(['html', 'xml']),
            )
        else:
            self.jinja_env = None
            self.logger.warning("report_builder.no_templates", path=str(self.template_dir))

    def _default_template_dir(self) -> Path:
        """Get default template directory."""
        return Path(__file__).parent / "templates"

    def write_bundle(self, state: InvestigationState) -> dict[str, Path]:
        """Write all report artifacts and return paths."""
        reports_dir = settings.runtime.reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        slug = self._slugify(state.subject)
        base_name = f"{timestamp}_{slug}"

        artifacts: dict[str, Path] = {}

        json_path = reports_dir / f"{base_name}.json"
        self._write_json_report(state, json_path)
        artifacts["json"] = json_path


        md_path = reports_dir / f"{base_name}.md"
        self._write_markdown_report(state, md_path)
        artifacts["markdown"] = md_path

        self.logger.info("report_builder.bundle_complete", artifacts=list(artifacts.keys()))
        return artifacts

    def _write_json_report(self, state: InvestigationState, path: Path) -> None:
        """Write structured JSON report."""
        payload = {
            "subject": state.subject,
            "objectives": state.objectives,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds") + "Z",
            "summary": self._build_summary(state),
            "findings": [
                {
                    "title": f.title,
                    "url": f.url,
                    "snippet": f.snippet[:200],
                    "confidence": f.confidence,
                }
                for f in state.findings
            ],
            "validated_facts": state.context.get("validated_facts", []),
            "risk_assessment": state.context.get("risk_summary", {}),
            "connections": state.context.get("connection_graph", {}),
            "leads": state.leads,
        }

        path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        self.logger.debug("report_builder.json_written", path=str(path))

    def _write_markdown_report(self, state: InvestigationState, path: Path) -> None:
        """Write human-readable markdown report."""
        sections = [
            f"# Risk Assessment Report: {state.subject}",
            f"\n**Generated:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Objectives:** {', '.join(state.objectives)}",
            "\n## Executive Summary",
            self._build_summary(state),
            "\n## Key Findings",
            self._format_findings(state),
            "\n## Risk Assessment",
            self._format_risks(state),
            "\n## Validated Facts",
            self._format_facts(state),
            "\n## Connection Analysis",
            self._format_connections(state),
            "\n## Recommended Actions",
            self._format_recommendations(state),
        ]

        path.write_text("\n".join(sections), encoding="utf-8")
        self.logger.debug("report_builder.markdown_written", path=str(path))

    def _build_summary(self, state: InvestigationState) -> str:
        """Build executive summary."""
        findings_count = len(state.findings)
        risks_count = len(state.risks)
        validated = len(state.context.get("validated_facts", []))
        risk_summary = state.context.get("risk_summary", {})
        overall_score = risk_summary.get("overall_score", 0.0)
        highest_severity = risk_summary.get("highest_severity", "none")

        return (
            f"Investigation of **{state.subject}** yielded {findings_count} findings, "
            f"with {validated} validated facts. Risk assessment identified {risks_count} potential "
            f"risk indicators with an overall score of {overall_score:.2f} and peak severity: {highest_severity}."
        )

    def _format_findings(self, state: InvestigationState) -> str:
        """Format findings section."""
        if not state.findings:
            return "- No findings available."

        lines = []
        for idx, finding in enumerate(state.findings[:10], 1):
            lines.append(f"{idx}. **{finding.title}**")
            lines.append(f"   - URL: {finding.url}")
            lines.append(f"   - Snippet: {finding.snippet[:150]}...")
            lines.append("")

        return "\n".join(lines)

    def _format_risks(self, state: InvestigationState) -> str:
        """Format risk assessment section."""
        risk_signals = state.context.get("risk_signals", [])

        if not risk_signals:
            return "- No specific risk signals identified."

        lines = []
        for signal in risk_signals[:10]:
            label = signal.get("label", "Unknown")
            category = signal.get("category", "general")
            severity = signal.get("severity", "unknown")
            confidence = signal.get("confidence", "unknown")
            lines.append(f"- **{label}** [{category}]")
            lines.append(f"  - Severity: {severity} | Confidence: {confidence}")
            lines.append(f"  - {signal.get('rationale', 'No rationale provided')}")
            lines.append("")

        return "\n".join(lines)

    def _format_facts(self, state: InvestigationState) -> str:
        """Format validated facts section."""
        facts = state.context.get("validated_facts", [])

        if not facts:
            return "- No validated facts available."

        lines = []
        for fact in facts[:15]:
            fact_text = fact.get("fact", "")
            confidence = fact.get("confidence", "unknown")
            support = fact.get("support_mentions", 0)
            lines.append(f"- {fact_text}")
            lines.append(f"  - Confidence: {confidence} | Support: {support} mentions")
            lines.append("")

        return "\n".join(lines)

    def _format_connections(self, state: InvestigationState) -> str:
        """Format connection analysis section."""
        graph = state.context.get("connection_graph", {})

        if not graph:
            return "- No connection data available."

        nodes = len(graph.get("nodes", []))
        edges = len(graph.get("edges", []))
        density = graph.get("density", 0)

        # Safe formatting to prevent None errors
        if density is not None:
            density_str = f"{density:.3f}"
        else:
            density_str = "N/A"

        lines = [
            f"- **Network Size:** {nodes} entities, {edges} connections",
            f"- **Density:** {density_str}",
            "",
            "**Key Entities (by centrality):**",
        ]

        centrality = graph.get("centrality", {})
        if centrality:
            top_entities = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            for name, score in top_entities:
                lines.append(f"- {name}: {score:.3f}")
        else:
            lines.append("- No centrality data available")

        return "\n".join(lines)

    def _format_recommendations(self, state: InvestigationState) -> str:
        """Format recommended next actions."""
        leads = state.leads[:5]

        if not leads:
            return "- No specific recommendations at this time."

        lines = ["Based on current findings, consider investigating:"]
        for lead in leads:
            lines.append(f"- {lead}")

        return "\n".join(lines)

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert text to filesystem-safe slug."""
        cleaned = "".join(c if c.isalnum() else "_" for c in text.lower())
        return "_".join(filter(None, cleaned.split("_")))
