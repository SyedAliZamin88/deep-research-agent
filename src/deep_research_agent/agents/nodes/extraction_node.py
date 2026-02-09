from __future__ import annotations
from typing import Any
import orjson
from deep_research_agent.core.state import InvestigationState
from deep_research_agent.llm_providers import ProviderType, create_provider
from .base import BaseAgentNode
class ExtractionNode(BaseAgentNode):
    """Uses an LLM to extract facts, leads, and risks from accumulated findings."""

    def __init__(self, provider: ProviderType, window_size: int = 8) -> None:
        super().__init__("extraction")
        self.provider = create_provider(provider)
        self.window_size = window_size

    async def run(self, state: InvestigationState) -> InvestigationState:
        findings_window = state.findings[-4:]
        if not findings_window:
            self.log("extraction.no_findings")
            state.record_log("extraction.no_findings", {})
            return state

        self.log("extraction.start", window=len(findings_window))
        prompt = self._compose_prompt(state, findings_window)

        async def invoke_coroutine():
            import traceback

            self.log("extraction.invoke_coroutine_start")
            try:
                res = await self.provider.invoke(
                    prompt,
                    system_prompt=(
                        "You are a due-diligence analyst. "
                        "Return strict JSON with keys facts, leads, risks, connections."
                    ),
                    temperature=0.05,
                    max_tokens=500,
                )
                self.log("extraction.invoke_coroutine_end", snippet=res[:200])
                return res
            except Exception as exc:
                self.log("extraction.invoke_inner_exception", error=str(exc), traceback=traceback.format_exc())
                raise

        try:
            import asyncio

            self.log("extraction.invoke_start")
            result = await asyncio.wait_for(invoke_coroutine(), timeout=20.0)
            self.log("extraction.invoke_completed")
        except asyncio.TimeoutError:
            self.log("extraction.invoke_timeout")
            state.record_log("extraction.invoke_timeout", {})
            return state
        except Exception as exc:
            import traceback

            self.log("extraction.invoke_error", error=str(exc), traceback=traceback.format_exc())
            state.record_log("extraction.invoke_error", {"error": str(exc), "traceback": traceback.format_exc()})
            return state

        raw_response = result
        if not raw_response:
            self.log("extraction.empty_response")
            state.record_log("extraction.empty_response", {})
            return state

        self.log("extraction.raw_response", raw=raw_response[:500])
        parsed = self._parse_response(raw_response)

        for fact in parsed["facts"]:
            state.record_log("extraction.fact", {"fact": fact})
        for lead in parsed["leads"]:
            state.add_lead(lead)
        for risk in parsed["risks"]:
            state.add_risk(risk)
        for connection in parsed["connections"]:
            state.add_connection(connection)

        state.context.setdefault("extracted_facts", []).extend(parsed["facts"])
        state.context.setdefault("identified_leads", []).extend(parsed["leads"])
        state.context.setdefault("identified_risks", []).extend(parsed["risks"])

        self.log(
            "extraction.complete",
            facts=len(parsed["facts"]),
            leads=len(parsed["leads"]),
            risks=len(parsed["risks"]),
        )
        return state

    def _compose_prompt(self, state: InvestigationState, findings: list[Any]) -> str:
        entries = "\n".join(
            f"- {artifact.title} :: {artifact.snippet[:200]} (source: {artifact.url})"
            for artifact in findings
        )
        return (
            f"Subject: {state.subject}\n"
            f"Context findings:\n{entries}\n"
            "Return JSON with keys:\n"
            "facts: list of strings describing verifiable facts.\n"
            "leads: list of follow-up search phrases or entities.\n"
            "risks: list of potential risk indicators.\n"
            "connections: list of objects describing relationships.\n"
        )

    def _parse_response(self, raw: str) -> dict[str, list[Any]]:
        try:
            cleaned = raw.strip()
            if cleaned.startswith('```'):
                cleaned = cleaned.split('\n', 1)[1] if '\n' in cleaned else cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned.rsplit('```', 1)[0]
            cleaned = cleaned.strip()

            payload = orjson.loads(cleaned)
        except orjson.JSONDecodeError:
            self.log("extraction.parse_error", raw=raw[:200])
            return {"facts": [], "leads": [], "risks": [], "connections": []}

        return {
            "facts": list(payload.get("facts", [])),
            "leads": list(payload.get("leads", [])),
            "risks": list(payload.get("risks", [])),
            "connections": list(payload.get("connections", [])),
        }
