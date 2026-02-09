# scripts/run_agent.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from deep_research_agent.agents.langgraph import ResearchGraphConfig
from deep_research_agent.core.orchestrator import ResearchOrchestrator
from deep_research_agent.utils import configure_logging, get_logger


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Deep Research Agent end-to-end on a given subject.",
    )
    parser.add_argument(
        "subject",
        help="Primary subject (individual or entity) to investigate.",
    )
    parser.add_argument(
        "--objective",
        action="append",
        dest="objectives",
        help="Add an investigation objective. May be provided multiple times.",
    )
    parser.add_argument(
        "--objectives-file",
        type=Path,
        help="Path to a text file with one objective per line.",
    )
    parser.add_argument(
        "--planner-provider",
        choices=["openai", "gemini", "ollama"],
        help="Override the LLM provider for the planning stage.",
    )
    parser.add_argument(
        "--extraction-provider",
        choices=["openai", "gemini", "ollama"],
        help="Override the LLM provider for the extraction stage.",
    )
    parser.add_argument(
        "--reporting-provider",
        choices=["openai", "gemini", "ollama"],
        help="Override the LLM provider for the reporting stage.",
    )
    parser.add_argument(
        "--search-provider",
        choices=["tavily", "serpapi", "web"],
        help="Override the search provider used for web discovery.",
    )
    parser.add_argument(
        "--search-results-per-query",
        type=int,
        default=None,
        help="Maximum results to keep from each search query.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum LangGraph passes when new leads are discovered.",
    )
    return parser.parse_args(list(argv))


def load_objectives(args: argparse.Namespace) -> list[str]:
    objectives: list[str] = []

    if args.objectives_file:
        if not args.objectives_file.exists():
            raise FileNotFoundError(f"Objectives file not found: {args.objectives_file}")
        text = args.objectives_file.read_text(encoding="utf-8")
        objectives.extend(line.strip() for line in text.splitlines() if line.strip())

    if args.objectives:
        objectives.extend(obj for obj in args.objectives if obj.strip())

    return [obj for obj in objectives if obj]


def build_config(args: argparse.Namespace) -> ResearchGraphConfig | None:
    overrides: dict[str, object] = {}

    if args.planner_provider:
        overrides["planner_provider"] = args.planner_provider
    if args.extraction_provider:
        overrides["extraction_provider"] = args.extraction_provider
    if args.reporting_provider:
        overrides["reporting_provider"] = args.reporting_provider
    if args.search_provider:
        overrides["search_provider"] = args.search_provider
    if args.search_results_per_query is not None:
        overrides["search_results_per_query"] = max(1, args.search_results_per_query)
    if args.max_iterations is not None:
        overrides["max_iterations"] = max(1, args.max_iterations)

    if not overrides:
        return None
    return ResearchGraphConfig(**overrides)


def format_summary(state) -> str:
    report_path = state.context.get("report_path", "N/A")
    summary_lines = [
        f"Subject: {state.subject}",
        f"Objectives: {len(state.objectives)}",
        f"Findings: {len(state.findings)}",
        f"Leads: {len(state.leads)}",
        f"Risks: {len(state.risks)}",
        f"Report saved to: {report_path}",
    ]
    return "\n".join(summary_lines)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    configure_logging()
    logger = get_logger(__name__)

    try:
        objectives = load_objectives(args)
    except Exception as exc:  # pragma: no cover - CLI utility
        logger.error("agent.cli.objectives_error", error=str(exc))
        return 1

    if not objectives:
        logger.error("agent.cli.no_objectives", message="Provide --objective or --objectives-file.")
        return 1

    config = build_config(args)
    orchestrator = ResearchOrchestrator(config=config)

    try:
        state = orchestrator.run(args.subject, objectives)
    except Exception as exc:  # pragma: no cover - top-level guard
        logger.error("agent.cli.runtime_error", error=str(exc))
        return 1

    logger.info("agent.cli.summary", summary=format_summary(state))
    print(format_summary(state))

    report_draft = state.context.get("report_draft")
    if report_draft:
        separator = "-" * 80
        print(f"\n{separator}\nRISK ASSESSMENT DRAFT\n{separator}\n{report_draft}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
