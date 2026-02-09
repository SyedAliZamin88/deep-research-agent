from __future__ import annotations
from typing import Any

from deep_research_agent.core.state import InvestigationState
from deep_research_agent.llm_providers import ProviderType, create_provider
from deep_research_agent.search import QueryPlanner
from .base import BaseAgentNode

class PlannerNode(BaseAgentNode):
    """Generates the initial investigation plan and starting query set."""

    def __init__(self, provider: ProviderType) -> None:
        super().__init__("planner")
        self.provider = create_provider(provider)
        self.query_planner = QueryPlanner()

    def run(self, state: InvestigationState) -> InvestigationState:
        self.log("planner.start", subject=state.subject)
        prompt = self._compose_prompt(state)

        def _invoke() -> str:
            return self._await_result(
                self.provider.invoke(
                    prompt,
                    system_prompt=(
                        "You are an elite risk-intelligence analyst. "
                        "Produce structured investigation plans with actionable next steps."
                    ),
                    temperature=0.2,
                )
            )

        plan_text = self.retry_policy.wrap(_invoke)()
        state.context["investigation_plan"] = plan_text

        initial_queries = self.query_planner.initial_queries(state)
        state.context["initial_queries"] = initial_queries

        self.log(
            "planner.complete",
            query_count=len(initial_queries),
            plan_excerpt=plan_text[:200],
        )
        return state

    def _compose_prompt(self, state: InvestigationState) -> str:
        objectives = "\n".join(f"- {obj}" for obj in state.objectives)
        return (
            "Create a concise, step-by-step investigation plan.\n"
            f"Subject: {state.subject}\n"
            f"Objectives:\n{objectives}\n"
            "Outline key focus areas, immediate queries, and data sources to pursue."
        )
