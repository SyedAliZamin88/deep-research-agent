from __future__ import annotations
from typing import Any
from deep_research_agent.core.state import InvestigationState, ResearchArtifact
from deep_research_agent.search import SearchProviderType, create_search_provider
from .base import BaseAgentNode

class SearchNode(BaseAgentNode):
    """Executes search queries and aggregates findings."""

    def __init__(self, provider: SearchProviderType, max_results: int = 6) -> None:
        super().__init__("search")
        self.provider = create_search_provider(provider)
        self.max_results = max_results

    def run(self, state: InvestigationState) -> InvestigationState:
        queries: list[str] = state.context.get("initial_queries", [])
        if not queries:
            self.log("search.no_queries")
            return state

        self.log("search.start", query_count=len(queries))
        for query in queries:
            self.log("search.executing", query=query)

            def _search() -> list[dict[str, Any]]:
                return self._await_result(self.provider.search(query, num_results=self.max_results))

            try:
                results = self.retry_policy.wrap(_search)()
                if results is None or not results:
                    self.log("search.no_results", query=query)
                    state.record_log("search.no_results", {"query": query})
                    continue
            except Exception as exc:  # broad, but logged and recorded
                self.log("search.error", query=query, error=str(exc))
                state.record_log("search.error", {"query": query, "error": str(exc)})
                continue

            for entry in results:
                artifact = ResearchArtifact(
                    title=entry.get("title") or "untitled",
                    url=entry.get("url") or "",
                    snippet=entry.get("snippet") or "",
                    metadata=entry.get("metadata") or {},
                )
                state.add_finding(artifact)

            self.log("search.completed", query=query, result_count=len(results))

        state.record_log("search.summary", {"total_findings": len(state.findings)})

        self.log("search.run_complete", total_findings=len(state.findings))

        return state
