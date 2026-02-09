from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence
from deep_research_agent.analysis.fact_validator import ValidationResult

@dataclass(slots=True)
class RiskSignal:
    """Normalized description of an identified risk indicator."""

    label: str
    category: str
    severity: str
    confidence: str
    rationale: str
    supporting_facts: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "category": self.category,
            "severity": self.severity,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "supporting_facts": self.supporting_facts,
            "sources": self.sources,
        }

@dataclass(slots=True)
class RiskScoreSummary:
    """Roll-up of risk metrics for reporting and downstream reasoning."""

    overall_score: float
    highest_severity: str
    category_breakdown: dict[str, float]
    signals: list[RiskSignal]

    def to_dict(self) -> dict[str, object]:
        return {
            "overall_score": round(self.overall_score, 3),
            "highest_severity": self.highest_severity,
            "category_breakdown": {
                category: round(score, 3) for category, score in self.category_breakdown.items()
            },
            "signals": [signal.to_dict() for signal in self.signals],
        }


DEFAULT_CATEGORY_KEYWORDS: Mapping[str, Sequence[str]] = {
    "legal": ("lawsuit", "litigation", "fraud", "indictment", "regulatory"),
    "financial": ("bankruptcy", "insolvency", "money laundering", "tax", "embezzle"),
    "reputation": ("controversy", "scandal", "misconduct", "criticism"),
    "security": ("breach", "hack", "espionage", "leak"),
    "compliance": ("sanction", "violation", "non-compliance"),
}

SEVERITY_WEIGHT = {"critical": 1.0, "high": 0.75, "medium": 0.5, "low": 0.25}
CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.6, "low": 0.35, "none": 0.1}


def classify_risk_category(risk_text: str, keywords: Mapping[str, Sequence[str]] | None = None) -> str:
    keywords = keywords or DEFAULT_CATEGORY_KEYWORDS
    text = risk_text.lower()
    best_match = "general"
    best_score = 0
    for category, tokens in keywords.items():
        score = sum(1 for token in tokens if token in text)
        if score > best_score:
            best_match = category
            best_score = score
    return best_match


def infer_severity(risk_text: str) -> str:
    text = risk_text.lower()
    if any(term in text for term in ("arrest", "felony", "major breach", "criminal charge")):
        return "critical"
    if any(term in text for term in ("regulator", "lawsuit", "whistleblower", "fraud", "sanction")):
        return "high"
    if any(term in text for term in ("investigation", "audit", "controversy", "breach", "dispute")):
        return "medium"
    return "low"


def score_signal(severity: str, confidence: str, amplifiers: float | None = None) -> float:
    base = SEVERITY_WEIGHT.get(severity, 0.2) * CONFIDENCE_WEIGHT.get(confidence, 0.2)
    if amplifiers is not None:
        base *= 1.0 + amplifiers
    return min(round(base, 3), 1.0)


def build_risk_signals(
    raw_risks: Iterable[str],
    validated_facts: Sequence[ValidationResult] | None = None,
    *,
    keywords: Mapping[str, Sequence[str]] | None = None,
) -> list[RiskSignal]:
    validated_fact_lookup: dict[str, ValidationResult] = {
        result.fact: result for result in validated_facts or []
    }

    signals: list[RiskSignal] = []
    for risk in raw_risks:
        if not risk or not risk.strip():
            continue

        category = classify_risk_category(risk, keywords)
        severity = infer_severity(risk)
        confidence = "low"
        supporting_facts: list[str] = []
        source_refs: list[str] = []

        if validated_fact := validated_fact_lookup.get(risk):
            confidence = validated_fact.confidence
            supporting_facts.append(validated_fact.fact)
            source_refs.extend(record.url for record in validated_fact.evidence)

        if not supporting_facts and validated_facts:
            for result in validated_facts:
                if any(token in result.normalized_fact for token in risk.lower().split()):
                    supporting_facts.append(result.fact)
                    source_refs.extend(record.url for record in result.evidence)
                    confidence = max(confidence, result.confidence, key=lambda key: CONFIDENCE_WEIGHT.get(key, 0))

        rationale = f"Identified as {category} risk with {severity} severity."
        if supporting_facts:
            rationale += f" Supported by {len(supporting_facts)} validated fact(s)."

        signals.append(
            RiskSignal(
                label=risk.strip(),
                category=category,
                severity=severity,
                confidence=confidence,
                rationale=rationale,
                supporting_facts=supporting_facts,
                sources=list(dict.fromkeys(source_refs)),
            )
        )
    return signals


def aggregate_risk_scores(signals: Sequence[RiskSignal]) -> RiskScoreSummary:
    if not signals:
        return RiskScoreSummary(
            overall_score=0.0,
            highest_severity="none",
            category_breakdown={},
            signals=[],
        )

    category_totals: dict[str, float] = {}
    highest_severity = "low"
    highest_score = 0.0
    overall = 0.0

    for signal in signals:
        score = score_signal(signal.severity, signal.confidence)
        overall += score
        category_totals[signal.category] = category_totals.get(signal.category, 0.0) + score

        if score > highest_score:
            highest_score = score
            highest_severity = signal.severity

    normalized_overall = min(round(overall / max(len(signals), 1), 3), 1.0)

    for category, total in category_totals.items():
        category_totals[category] = min(round(total, 3), 1.0)

    return RiskScoreSummary(
        overall_score=normalized_overall,
        highest_severity=highest_severity,
        category_breakdown=dict(sorted(category_totals.items(), key=lambda item: item[1], reverse=True)),
        signals=list(signals),
    )


__all__ = [
    "RiskSignal",
    "RiskScoreSummary",
    "build_risk_signals",
    "aggregate_risk_scores",
    "classify_risk_category",
    "infer_severity",
]
