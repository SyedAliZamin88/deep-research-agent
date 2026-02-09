"""Custom exceptions for the deep research agent."""

class DeepResearchError(Exception):
    """Base exception for all agent errors."""
    pass

class ProviderNotConfiguredError(DeepResearchError):
    """Raised when a required provider (LLM/search) is not configured."""
    pass

class SearchQueryError(DeepResearchError):
    """Raised when a search query fails."""
    pass

class ValidationError(DeepResearchError):
    """Raised when validation fails."""
    pass

class StateError(DeepResearchError):
    """Raised when state management encounters an error."""
    pass
