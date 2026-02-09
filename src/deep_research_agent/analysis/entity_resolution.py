from __future__ import annotations
import math
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable, Mapping, Sequence

WordWeight = Mapping[str, float]

@dataclass(slots=True)
class EntityCandidate:
    """Represents a canonical entity derived from multiple raw mentions."""

    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    score: float = 0.0
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "canonical_name": self.canonical_name,
            "aliases": self.aliases,
            "score": round(self.score, 3),
            "sources": self.sources,
            "metadata": self.metadata,
        }


DEFAULT_STOP_WORDS = {
    "inc",
    "llc",
    "ltd",
    "corp",
    "co",
    "company",
    "group",
    "plc",
    "the",
    "and",
    "&",
}


def normalize_name(name: str) -> str:
    """Normalize an entity name for consistent comparisons."""
    normalized = unicodedata.normalize("NFKD", name)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9\\s]", " ", normalized)
    tokens = [token for token in normalized.split() if token and token not in DEFAULT_STOP_WORDS]
    return " ".join(tokens)


def _tokenize(name: str) -> list[str]:
    normalized = normalize_name(name)
    return [token for token in normalized.split() if token]


def _initials(tokens: Sequence[str]) -> str:
    return "".join(token[0] for token in tokens if token)


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _token_overlap(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def score_alias_similarity(alias: str, candidate: str) -> float:
    tokens_alias = _tokenize(alias)
    tokens_candidate = _tokenize(candidate)
    base = _similarity(normalize_name(alias), normalize_name(candidate))
    token_overlap = _token_overlap(tokens_alias, tokens_candidate)
    initials_score = 1.0 if _initials(tokens_alias) == _initials(tokens_candidate) else 0.0
    score = (base * 0.6) + (token_overlap * 0.3) + (initials_score * 0.1)
    return round(min(score, 1.0), 3)


def generate_aliases(name: str) -> list[str]:
    tokens = _tokenize(name)
    if not tokens:
        return []

    aliases = {name.strip()}
    if len(tokens) >= 2:
        aliases.add(" ".join(tokens[:2]))
        aliases.add(" ".join(tokens[-2:]))
    aliases.add(_initials(tokens))
    aliases.update(tokens)
    return [alias for alias in aliases if alias]


def _word_weight_lookup(tokens: Sequence[str], weights: WordWeight | None = None) -> float:
    if not tokens:
        return 1.0
    weights = weights or {}
    total = sum(weights.get(token, 1.0) for token in tokens)
    return total / len(tokens)


def resolve_entities(
    mentions: Iterable[str],
    *,
    threshold: float = 0.68,
    weights: WordWeight | None = None,
    source_index: Mapping[str, list[str]] | None = None,
) -> list[EntityCandidate]:
    """Resolve raw mentions into canonical entities using heuristic clustering."""
    normalized_map: dict[str, list[str]] = {}
    for mention in mentions:
        if not mention or not mention.strip():
            continue
        normalized = normalize_name(mention)
        if not normalized:
            continue
        normalized_map.setdefault(normalized, []).append(mention.strip())

    if not normalized_map:
        return []

    clusters: list[list[str]] = []
    processed: set[str] = set()
    normalized_items = list(normalized_map.keys())
    for normalized in normalized_items:
        if normalized in processed:
            continue
        cluster = {normalized}
        processed.add(normalized)

        for other in normalized_items:
            if other in processed:
                continue
            similarity = _similarity(normalized, other)
            token_overlap = _token_overlap(normalized.split(), other.split())
            combined_score = (similarity * 0.7) + (token_overlap * 0.3)
            if combined_score >= threshold:
                cluster.add(other)
                processed.add(other)
        clusters.append(list(cluster))

    resolved_candidates: list[EntityCandidate] = []
    for cluster in clusters:
        original_mentions: list[str] = []
        for key in cluster:
            original_mentions.extend(normalized_map.get(key, []))

        tokens = _tokenize(original_mentions[0])
        canonical = " ".join(tokens) if tokens else original_mentions[0]
        alias_set = set()
        for mention in original_mentions:
            alias_set.update(generate_aliases(mention))
        alias_set.update(generate_aliases(canonical))
        alias_set.discard("")
        aliases = sorted(alias_set)

        weights_score = _word_weight_lookup(tokens, weights)

        similarity_scores = [
            score_alias_similarity(mention, canonical) for mention in original_mentions
        ]
        average_similarity = sum(similarity_scores) / max(len(similarity_scores), 1)
        total_score = min(round(math.sqrt(weights_score * average_similarity), 3), 1.0)
        candidate_sources: list[str] = []
        if source_index:
            for mention in original_mentions:
                candidate_sources.extend(source_index.get(mention, []))
        candidate_sources = list(dict.fromkeys(candidate_sources))
        resolved_candidates.append(
            EntityCandidate(
                canonical_name=canonical,
                aliases=aliases,
                score=total_score,
                sources=candidate_sources,
                metadata={
                    "mention_count": len(original_mentions),
                    "average_similarity": round(average_similarity, 3),
                },
            )
        )
    resolved_candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return resolved_candidates


def build_entity_index(candidates: Sequence[EntityCandidate]) -> dict[str, EntityCandidate]:
    """Create a lookup dictionary keyed by canonical name and alias."""
    index: dict[str, EntityCandidate] = {}
    for candidate in candidates:
        index[candidate.canonical_name] = candidate
        for alias in candidate.aliases:
            index.setdefault(alias, candidate)
    return index


__all__ = [
    "EntityCandidate",
    "normalize_name",
    "score_alias_similarity",
    "generate_aliases",
    "resolve_entities",
    "build_entity_index",
]
