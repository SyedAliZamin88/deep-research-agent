import json
import sys
from datetime import datetime, timezone as UTC
from pathlib import Path

from deep_research_agent.core.orchestrator import ResearchOrchestrator
from deep_research_agent.core.state import InvestigationState
from deep_research_agent.utils import configure_logging, get_logger


class PersonaEvaluator:
    """Evaluates agent performance against a test persona with known hidden facts."""

    def __init__(self, persona_path: Path) -> None:
        self.persona_path = persona_path
        self.persona = self._load_persona()
        self.logger = get_logger(__name__)

    def _load_persona(self) -> dict:
        if not self.persona_path.exists():
            raise FileNotFoundError(f"Persona file not found: {self.persona_path}")
        with open(self.persona_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def run_investigation(self) -> InvestigationState:
        self.logger.info(
            "evaluator.start",
            persona_id=self.persona.get('id'),
            name=self.persona.get('name')
        )
        objectives = self.persona.get('evaluation_targets', {}).get('must_surface', [])
        if not objectives:
            objectives = [
                "Surface hidden ownership structures",
                "Identify sanction exposure",
                "Map suspicious connections"
            ]
        orchestrator = ResearchOrchestrator()
        raw_state = orchestrator.run(
            subject=self.persona.get('name'),
            objectives=objectives
        )
        if isinstance(raw_state, dict):
            state = InvestigationState(
                subject=raw_state.get('subject', self.persona.get('name')),
                objectives=raw_state.get('objectives', objectives),
                findings=raw_state.get('findings', []),
                risks=raw_state.get('risks', []),
                connections=raw_state.get('connections', []),
                context=raw_state.get('context', {})
            )
        else:
            state = raw_state
        self.logger.info(
            "evaluator.complete",
            findings=len(state.findings),
            facts=len(state.context.get('extracted_facts', [])),
            risks=len(state.risks)
        )
        return state

    def evaluate_results(self, state: InvestigationState) -> dict:
        persona = self.persona
        objectives = persona.get('evaluation_targets', {}).get('must_surface', []) or []
        hidden_facts = persona.get('hidden_facts', []) or []
        risk_signals = persona.get('risk_signals', []) or []
        expected_connections = persona.get('connections', []) or []

        findings_text = self._gather_findings_text(state.findings if hasattr(state, 'findings') else [])
        facts_text = self._gather_text(state.context.get('extracted_facts', []))
        risks_text = ' '.join([r.get('description', '').lower() for r in risk_signals])
        connections = state.connections if hasattr(state, 'connections') else []

        coverage_score = self._score_coverage(objectives, findings_text)
        fact_score = self._score_hidden_facts(hidden_facts, facts_text)
        risk_score = self._score_risks(risk_signals, risks_text)
        connection_score = self._score_connections(expected_connections, connections)

        weights = persona.get('scoring_guidance', {})
        coverage_w = weights.get('coverage_weight', 0.4)
        accuracy_w = weights.get('accuracy_weight', 0.35)
        confidence_w = weights.get('confidence_weight', 0.15)
        presentation_w = weights.get('presentation_weight', 0.1)

        total_score = (
            coverage_w * coverage_score
            + accuracy_w * fact_score
            + confidence_w * risk_score
            + presentation_w * connection_score
        )

        results = {
            'scores': {
                'must_surface': coverage_score,
                'hidden_facts': fact_score,
                'risk_identification': risk_score,
                'connection_mapping': connection_score,
                'total_score': total_score,
            },
            'metrics': {
                'findings_count': len(state.findings) if hasattr(state, 'findings') else 0,
                'facts_extracted': len(state.context.get('extracted_facts', [])),
                'risks_identified': len(state.risks) if hasattr(state, 'risks') else 0,
                'connections_mapped': len(state.connections) if hasattr(state, 'connections') else 0,
            },
            'pass_threshold': weights.get('pass_threshold', 0.75),
            'pass': total_score >= weights.get('pass_threshold', 0.75),
            'persona_id': persona.get('id'),
            'persona_name': persona.get('name'),
        }

        return results

    def _gather_text(self, facts: list) -> str:
        text_parts = []
        for fact in facts:
            if isinstance(fact, dict):
                text_parts.append(str(fact.get('fact', '')))
                text_parts.append(str(fact.get('normalized_fact', '')))
            else:
                text_parts.append(str(fact))
        return ' '.join(text_parts).lower()

    def _gather_findings_text(self, findings: list) -> str:
        text_parts = []
        for finding in findings:
            text_parts.append(finding.title if hasattr(finding, 'title') else '')
            text_parts.append(finding.snippet if hasattr(finding, 'snippet') else '')
        return ' '.join(text_parts).lower()

    def _gather_connections_text(self, connections: list) -> str:
        text_parts = []
        for conn in connections:
            text_parts.append(str(conn.get('source', '')))
            text_parts.append(str(conn.get('target', '')))
            text_parts.append(str(conn.get('relation', '')))
            text_parts.append(str(conn.get('notes', '')))
        return ' '.join(text_parts).lower()

    def _score_coverage(self, targets: list, content: str) -> float:
        if not targets:
            return 1.0

        found = 0
        for target in targets:
            key_terms = self._extract_key_terms(target)
            if any(term in content for term in key_terms):
                found += 1

        return found / len(targets)

    def _score_hidden_facts(self, hidden_facts: list, content: str) -> float:
        if not hidden_facts:
            return 1.0

        found = 0
        for fact in hidden_facts:
            label = fact.get('label', '').lower()
            description = fact.get('description', '').lower()

            key_terms = self._extract_key_terms(f"{label} {description}")
            if sum(1 for term in key_terms if term in content) >= len(key_terms) * 0.4:
                found += 1

        return found / len(hidden_facts)

    def _score_risks(self, expected_risks: list, risks_text: str) -> float:
        if not expected_risks:
            return 1.0

        found = 0
        for risk in expected_risks:
            category = risk.get('category', '').lower()
            description = risk.get('description', '').lower()

            if category in risks_text or any(word in risks_text for word in description.split()[:5]):
                found += 1

        return found / len(expected_risks)

    def _score_connections(self, expected: list, actual: list) -> float:
        if not expected:
            return 1.0

        found = 0
        actual_text = ' '.join(str(c) for c in actual).lower()

        for exp_conn in expected:
            source = str(exp_conn.get('source', '')).lower()
            target = str(exp_conn.get('target', '')).lower()

            if source in actual_text and target in actual_text:
                found += 1

        return found / len(expected)

    def _extract_key_terms(self, text: str) -> list:
        words = text.lower().split()
        key_terms = [w for w in words if len(w) > 3 and w not in {
            'that', 'with', 'from', 'this', 'have', 'been', 'were', 'their'
        }]
        return key_terms


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    logger = get_logger(__name__)

    argv = argv or sys.argv[1:]

    persona_dir = Path("data/personas")
    if not persona_dir.exists():
        logger.error("evaluator.no_personas", path=str(persona_dir))
        print(f"Error: Persona directory not found: {persona_dir}")
        return 1

    persona_files = list(persona_dir.glob("persona_*.json"))
    if not persona_files:
        logger.error("evaluator.no_files", path=str(persona_dir))
        print(f"Error: No persona files found in {persona_dir}")
        return 1

    logger.info("evaluator.start", persona_count=len(persona_files))
    print(f"\n{'='*80}")
    print(f"EVALUATING {len(persona_files)} PERSONAS")
    print(f"{'='*80}\n")

    all_results = {}

    for persona_file in sorted(persona_files):
        print(f"\n{'─'*80}")
        print(f"Evaluating: {persona_file.name}")
        print(f"{'─'*80}")

        try:
            evaluator = PersonaEvaluator(persona_file)
            print(f"Running investigation on '{evaluator.persona.get('name')}'...")
            state = evaluator.run_investigation()

            print("\nScoring results...")
            results = evaluator.evaluate_results(state)

            print(f"\n{'─'*40}")
            print("RESULTS:")
            print(f"{'─'*40}")
            print(f"Total Score: {results['scores']['total_score']:.1%}")
            print(f"  - Must Surface: {results['scores']['must_surface']:.1%}")
            print(f"  - Hidden Facts: {results['scores']['hidden_facts']:.1%}")
            print(f"  - Risk ID: {results['scores']['risk_identification']:.1%}")
            print(f"  - Connections: {results['scores']['connection_mapping']:.1%}")
            print(f"\nMetrics:")
            print(f"  - Findings: {results['metrics']['findings_count']}")
            print(f"  - Facts: {results['metrics']['facts_extracted']}")
            print(f"  - Risks: {results['metrics']['risks_identified']}")
            print(f"  - Connections: {results['metrics']['connections_mapped']}")
            print(f"\nStatus: {'✅ PASS' if results['pass'] else '❌ FAIL'}")

            all_results[persona_file.stem] = results

        except Exception as exc:
            import traceback
            logger.error("evaluator.error", persona=persona_file.name, error=str(exc))
            print(f"❌ Error evaluating {persona_file.name}: {exc}")
            print("\n🔍 FULL TRACEBACK:")
            traceback.print_exc()
            continue

    results_file = Path("data/evaluation_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    logger.info("evaluator.complete", total=total_personas, passed=passed, avg_score=avg_score)


if __name__ == "__main__":
    sys.exit(main())
