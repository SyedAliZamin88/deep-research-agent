from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

try:
    import networkx as nx
except Exception:
    nx = None

@dataclass(slots=True)
class GraphNode:
    """Representation of an entity within the connection graph."""

    name: str
    kind: str = "entity"
    count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "count": self.count,
            "metadata": self.metadata,
        }

@dataclass(slots=True)
class GraphEdge:
    """Representation of a relationship between two entities."""
    source: str
    target: str
    relation: str = "associated_with"
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": self.weight,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ConnectionGraphSummary:
    """Serializable summary of an entity-relationship graph."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    centrality: dict[str, float]
    density: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "centrality": {node: round(score, 4) for node, score in self.centrality.items()},
            "density": round(self.density, 4),
        }


def normalize_connection_payload(payload: Mapping[str, Any]) -> GraphEdge | None:
    """Coerce an arbitrary mapping into a GraphEdge instance when possible."""
    source = (payload.get("source") or payload.get("from") or "").strip()
    target = (payload.get("target") or payload.get("to") or "").strip()
    if not source or not target or source == target:
        return None

    relation = (payload.get("relation") or payload.get("type") or "associated_with").strip()
    weight_raw = payload.get("weight") or payload.get("score") or 1.0
    try:
        weight = float(weight_raw)
    except (TypeError, ValueError):
        weight = 1.0

    metadata = dict(payload.get("metadata") or {})
    evidence = payload.get("evidence")
    if evidence:
        metadata.setdefault("evidence", evidence)

    return GraphEdge(
        source=source,
        target=target,
        relation=relation or "associated_with",
        weight=max(weight, 0.0),
        metadata=metadata,
    )


def parse_connection_records(records: Iterable[Mapping[str, Any]]) -> list[GraphEdge]:
    """Parse raw connection dictionaries into GraphEdge objects."""
    parsed: list[GraphEdge] = []
    for record in records:
        edge = normalize_connection_payload(record)
        if edge:
            parsed.append(edge)
    return parsed


def _fallback_degree_centrality(edges: Sequence[GraphEdge]) -> dict[str, float]:
    if not edges:
        return {}
    degree: dict[str, int] = {}
    for edge in edges:
        degree[edge.source] = degree.get(edge.source, 0) + 1
        degree[edge.target] = degree.get(edge.target, 0) + 1

    max_degree = max(degree.values(), default=1)
    if max_degree <= 0:
        max_degree = 1

    return {node: round(count / max_degree, 4) for node, count in degree.items()}


def _fallback_density(nodes: Sequence[GraphNode], edges: Sequence[GraphEdge]) -> float:
    """
    CRITICAL FIX: Calculate graph density with division-by-zero protection.
    """
    n = len(nodes)
    if n <= 1:
        return 0.0

    possible_edges = n * (n - 1) / 2
    if possible_edges <= 0:
        return 0.0

    actual_edges = len({(min(edge.source, edge.target), max(edge.source, edge.target)) for edge in edges})
    return round(actual_edges / possible_edges, 4)


def build_connection_graph(
    connections: Iterable[Mapping[str, Any]],
    *,
    compute_metrics: bool = True,
    node_kinds: Mapping[str, str] | None = None,
) -> ConnectionGraphSummary:
    """Construct a connection graph summary from raw connection payloads."""
    edges = parse_connection_records(connections)
    node_map: dict[str, GraphNode] = {}
    for edge in edges:
        for endpoint in (edge.source, edge.target):
            node = node_map.setdefault(endpoint, GraphNode(name=endpoint))
            node.count += 1
            if node_kinds and endpoint in node_kinds:
                node.kind = node_kinds[endpoint]

    nodes = list(node_map.values())

    if not compute_metrics:
        return ConnectionGraphSummary(nodes=nodes, edges=edges, centrality={}, density=0.0)

    if nx is not None:
        graph = nx.Graph()
        for node in nodes:
            graph.add_node(node.name, kind=node.kind, count=node.count, metadata=node.metadata)
        for edge in edges:
            graph.add_edge(
                edge.source,
                edge.target,
                weight=edge.weight,
                relation=edge.relation,
                metadata=edge.metadata,
            )
        centrality = nx.degree_centrality(graph) if graph.number_of_nodes() else {}
        density = float(nx.density(graph)) if graph.number_of_nodes() > 1 else 0.0
    else:
        centrality = _fallback_degree_centrality(edges)
        density = _fallback_density(nodes, edges)

    return ConnectionGraphSummary(nodes=nodes, edges=edges, centrality=centrality, density=density)


def filter_graph_edges(edges: Sequence[GraphEdge], *, min_weight: float = 0.0, relation: str | None = None) -> list[GraphEdge]:
    """Filter graph edges by weight threshold and optional relation name."""
    filtered: list[GraphEdge] = []
    for edge in edges:
        if edge.weight < min_weight:
            continue
        if relation and edge.relation != relation:
            continue
        filtered.append(edge)
    return filtered


def summarize_connections(connections: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Produce a JSON-ready connection summary from raw connection payloads."""
    summary = build_connection_graph(connections)
    return summary.to_dict()

__all__ = [
    "GraphNode",
    "GraphEdge",
    "ConnectionGraphSummary",
    "normalize_connection_payload",
    "parse_connection_records",
    "build_connection_graph",
    "filter_graph_edges",
    "summarize_connections",
]
