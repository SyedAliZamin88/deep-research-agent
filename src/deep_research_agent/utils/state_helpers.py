"""Helper utilities for safe state operations and calculations."""

from __future__ import annotations
from typing import Any, Dict, Iterable, TypeVar
from deep_research_agent.core.state import InvestigationState
T = TypeVar('T', int, float)
def safe_divide(
    numerator: T,
    denominator: T,
    default: T | None = None
) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: The numerator
        denominator: The denominator
        default: Value to return if denominator is 0 (defaults to 0.0)

    Returns:
        Result of division or default value

    Examples:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0)
        0.0
        >>> safe_divide(10, 0, default=None)
        None
    """
    if denominator == 0:
        return default if default is not None else 0.0
    return numerator / denominator


def safe_percentage(
    part: T,
    whole: T,
    default: float = 0.0
) -> float:
    """
    Safely calculate percentage, handling zero denominator.

    Args:
        part: The part value
        whole: The whole value
        default: Value to return if whole is 0

    Returns:
        Percentage (0-100) or default

    Examples:
        >>> safe_percentage(25, 100)
        25.0
        >>> safe_percentage(1, 0)
        0.0
    """
    if whole == 0:
        return default
    return (part / whole) * 100


def safe_average(values: Iterable[float], default: float = 0.0) -> float:
    """
    Safely calculate average of values, handling empty list.

    Args:
        values: Iterable of numeric values
        default: Value to return if list is empty

    Returns:
        Average or default

    Examples:
        >>> safe_average([1, 2, 3])
        2.0
        >>> safe_average([])
        0.0
    """
    values_list = list(values)
    if not values_list:
        return default
    return sum(values_list) / len(values_list)


def ensure_investigation_state(state: InvestigationState | dict) -> InvestigationState:
    """
    Convert dict to InvestigationState if needed, or return existing InvestigationState.

    This handles the case where LangGraph might pass state as a dict instead of
    the proper InvestigationState dataclass.

    Args:
        state: Either InvestigationState object or dict

    Returns:
        Proper InvestigationState object

    Examples:
        >>> state_dict = {"subject": "John Doe", "objectives": ["research"]}
        >>> state = ensure_investigation_state(state_dict)
        >>> isinstance(state, InvestigationState)
        True
    """
    if isinstance(state, InvestigationState):
        return state

    return InvestigationState(
        subject=state.get("subject", ""),
        objectives=state.get("objectives", []),
        findings=state.get("findings", []),
        leads=state.get("leads", []),
        risks=state.get("risks", []),
        connections=state.get("connections", []),
        context=state.get("context", {}),
        logs=state.get("logs", [])
    )


def calculate_confidence_metrics(
    validated_count: int,
    total_count: int,
    high_confidence_count: int = 0
) -> Dict[str, float]:
    """
    Calculate validation and confidence metrics safely.

    Args:
        validated_count: Number of validated items
        total_count: Total number of items
        high_confidence_count: Number of high-confidence items

    Returns:
        Dict with validation_rate, high_confidence_rate, and overall_quality
    """
    validation_rate = safe_percentage(validated_count, total_count)
    high_conf_rate = safe_percentage(high_confidence_count, validated_count)
    overall_quality = (validation_rate + high_conf_rate) / 2 if validated_count > 0 else 0.0

    return {
        "validation_rate": validation_rate,
        "high_confidence_rate": high_conf_rate,
        "overall_quality": overall_quality,
        "total_items": total_count,
        "validated_items": validated_count,
        "high_confidence_items": high_confidence_count
    }


def format_metric(value: float | None, precision: int = 2, suffix: str = "") -> str:
    """
    Format a metric value safely, handling None.

    Args:
        value: The numeric value or None
        precision: Number of decimal places
        suffix: Optional suffix (e.g., '%', 'x')

    Returns:
        Formatted string

    Examples:
        >>> format_metric(0.12345, precision=2, suffix='%')
        '0.12%'
        >>> format_metric(None)
        'N/A'
    """
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}{suffix}"
