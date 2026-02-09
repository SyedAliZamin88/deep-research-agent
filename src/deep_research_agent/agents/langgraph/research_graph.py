from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List
from langgraph.graph import START, END, StateGraph
from deep_research_agent.agents.nodes import (
    ExtractionNode,
    PlannerNode,
    ReportingNode,
    SearchNode,
    ValidationNode,
)
from deep_research_agent.core.state import InvestigationState
from deep_research_agent.llm_providers import ProviderType
from deep_research_agent.search import QueryPlanner, SearchProviderType
from deep_research_agent.utils import get_logger

try:
    from deep_research_agent.observability.openinference import get_default_client
except Exception:
    get_default_client = None

@dataclass(slots=True)
class ResearchGraphConfig:
    """Configuration for assembling the LangGraph workflow."""

    planner_provider: ProviderType = "openai"
    search_provider: SearchProviderType = "tavily"
    extraction_provider: ProviderType = "openai"
    reporting_provider: ProviderType = "openai"
    search_results_per_query: int = 6
    max_iterations: int = 1  # Number of passes allowed for lead-driven refinement


class ResearchGraph:
    """Composed LangGraph workflow orchestrating the research agent nodes."""

    def __init__(self, config: ResearchGraphConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)
        self.query_planner = QueryPlanner()

        self._planner_node = PlannerNode(config.planner_provider)
        self._search_node = SearchNode(config.search_provider, max_results=config.search_results_per_query)
        self._extraction_node = ExtractionNode(config.extraction_provider)
        self._validation_node = ValidationNode()
        self._reporting_node = ReportingNode(config.reporting_provider)

        self._graph = self._build_graph()

    def run(self, state: InvestigationState) -> InvestigationState:
        """Execute the workflow, optionally iterating when new leads emerge."""
        self.logger.info(
            "graph.run.start",
            subject=state.subject,
            max_iterations=self.config.max_iterations,
        )

        current_state = state
        last_iteration = 0

        for iteration in range(1, self.config.max_iterations + 1):
            last_iteration = iteration

            # Safe context update
            if hasattr(current_state, 'context'):
                current_state.context["iteration"] = iteration
            else:
                if "context" not in current_state:
                    current_state["context"] = {}
                current_state["context"]["iteration"] = iteration

            # Safe record_log
            if hasattr(current_state, 'record_log'):
                current_state.record_log("graph.iteration_start", {"iteration": iteration})

            # Execute the graph
            current_state = self._graph.invoke(current_state)

            # Safe record_log
            if hasattr(current_state, 'record_log'):
                current_state.record_log("graph.iteration_end", {"iteration": iteration})

            if not self._should_continue(current_state, iteration):
                break

            self._prepare_next_iteration(current_state)

        self.logger.info(
            "graph.run.finish",
            iterations=last_iteration,
            findings=len(current_state.findings) if hasattr(current_state, 'findings') else len(current_state.get('findings', [])),
            leads=len(current_state.context.get("identified_leads", [])) if hasattr(current_state, 'context') else len(current_state.get("context", {}).get("identified_leads", [])),
        )

        return current_state

    def _build_graph(self) -> StateGraph:
        builder: StateGraph = StateGraph(InvestigationState)

        builder.add_node("planner", self._wrap(self._planner_node.run, "planner"))
        builder.add_node("search", self._wrap(self._search_node.run, "search"))
        builder.add_node("extraction", self._wrap(self._extraction_node.run, "extraction"))
        builder.add_node("validation", self._wrap(self._validation_node.run, "validation"))
        builder.add_node("reporting", self._wrap(self._reporting_node.run, "reporting"))

        builder.add_edge(START, "planner")
        builder.add_edge("planner", "search")
        builder.add_edge("search", "extraction")
        builder.add_edge("extraction", "validation")
        builder.add_edge("validation", "reporting")
        builder.add_edge("reporting", END)

        return builder.compile()

    @staticmethod
    def _wrap(func: Callable[[InvestigationState], InvestigationState], node_name: str | None = None) -> Callable[[InvestigationState], InvestigationState]:
        """
        Adapter so each node fits LangGraph's callable signature and emits an
        OpenInference span (when available). Design goals:
        - Non-fatal: if the OpenInference client is unavailable or errors, the
          wrapper simply runs the node normally.
        - Provides a clear span name `node.<node_name>` for tracing and a small
          structured event payload with iteration and subject when available.
        """
        import asyncio
        import inspect
        import time
        from typing import Optional

        oi_client_creator = None
        try:
            oi_client_creator = get_default_client
        except Exception:
            oi_client_creator = None

        def wrapped(state: InvestigationState) -> InvestigationState:
            span_ctx = None
            tracer = None

            try:
                if oi_client_creator:
                    tracer = oi_client_creator()
            except Exception:
                tracer = None

            span_name = f"node.{node_name or getattr(func, '__name__', 'node')}"
            iteration = None
            subject = None
            try:
                iteration = state.context.get("iteration") if hasattr(state, "context") else state.get("context", {}).get("iteration")
                subject = getattr(state, "subject", None) or (state.get("subject") if isinstance(state, dict) else None)
            except Exception:
                iteration = None
                subject = None

            if tracer:
                try:
                    span_ctx = tracer.span(span_name, attributes={"node": node_name, "subject": subject, "iteration": iteration})
                except Exception:
                    span_ctx = None

            start_ts = time.monotonic()
            if span_ctx is not None:
                try:
                    with span_ctx:
                        result = asyncio.run(func(state)) if inspect.iscoroutinefunction(func) else func(state)

                        try:
                            duration = time.monotonic() - start_ts
                            tracer.add_event(span_ctx._span if hasattr(span_ctx, "_span") else span_ctx, "node.completed", {"duration_s": duration})
                        except Exception:

                            pass
                        return result
                except Exception:
                    raise
            else:
                return asyncio.run(func(state)) if inspect.iscoroutinefunction(func) else func(state)

        return wrapped

    def _should_continue(self, state: InvestigationState | dict, iteration: int) -> bool:
        """Determine whether another pass should run."""
        if iteration >= self.config.max_iterations:
            return False

        pending = self._collect_new_leads(state)
        if not pending:
            return False

        if isinstance(state, dict):
            state.setdefault("context", {})["pending_leads"] = pending
        else:
            state.context["pending_leads"] = pending
        return True

    def _collect_new_leads(self, state: InvestigationState | dict) -> List[str]:
        """Identify leads that have not been processed yet."""
        if isinstance(state, dict):
            leads = list(state.get("context", {}).get("identified_leads", []))
            processed = state.setdefault("context", {}).setdefault("processed_leads", [])
        else:
            leads = list(state.context.get("identified_leads", []))
            processed = state.context.setdefault("processed_leads", [])
        return [lead for lead in leads if lead not in processed]

    def _prepare_next_iteration(self, state: InvestigationState | dict) -> None:
        """Update state with refined queries derived from new leads."""
        if isinstance(state, dict):
            pending_leads = state.get("context", {}).pop("pending_leads", [])
        else:
            pending_leads: List[str] = state.context.pop("pending_leads", [])
        if not pending_leads:
            return

        refined_queries = self.query_planner.refine_queries(state, pending_leads)
        if refined_queries:
            if isinstance(state, dict):
                state.setdefault("context", {})["initial_queries"] = refined_queries
            else:
                state.context["initial_queries"] = refined_queries

            if hasattr(state, 'record_log'):
                state.record_log(
                    "graph.prepare_next_iteration",
                    {"pending_leads": pending_leads, "refined_queries": refined_queries},
                )

        if isinstance(state, dict):
            processed = state.setdefault("context", {}).setdefault("processed_leads", [])
            processed.extend(pending_leads)
        else:
            processed = state.context.setdefault("processed_leads", [])
            processed.extend(pending_leads)
