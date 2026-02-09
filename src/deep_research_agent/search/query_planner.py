from __future__ import annotations
from typing import Any
from deep_research_agent.core.state import InvestigationState
from deep_research_agent.utils import get_logger

class QueryPlanner:
    """Generates search queries based on investigation state and objectives."""

    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    def initial_queries(self, state: InvestigationState | dict) -> list[str]:
        """Generate initial search queries from subject and objectives."""
        subject = state.subject if hasattr(state, 'subject') else state.get('subject', '')
        objectives = state.objectives if hasattr(state, 'objectives') else state.get('objectives', [])
        objectives_str = ", ".join(objectives)

        queries = [
            f"{subject} biography",
            f"{subject} controversy",
            f"{subject} financial connections",
            f"{subject} affiliations",
            f"{subject} risk factors {objectives_str}",
        ]

        self.logger.debug("query_planner.initial", queries=queries)
        return queries

    def refine_queries(self, state: InvestigationState | dict, new_leads: list[str]) -> list[str]:
        """Refine queries based on newly discovered leads."""
        subject = state.subject if hasattr(state, 'subject') else state.get('subject', '')
        refined: list[str] = []
        for lead in new_leads[:5]:  # Limit to avoid explosion
            refined.append(f"{subject} {lead} investigation")
            refined.append(f"{lead} relation to {subject}")

        self.logger.debug("query_planner.refined", new_queries=refined)
        return refined

    def focus_queries(self, state: InvestigationState, risk_domains: list[str]) -> list[str]:
        """Generate focused queries for specific risk domains."""
        focused = [f"{state.subject} {domain} risk" for domain in risk_domains]
        self.logger.debug("query_planner.focused", risk_domains=risk_domains, queries=focused)
        return focused
