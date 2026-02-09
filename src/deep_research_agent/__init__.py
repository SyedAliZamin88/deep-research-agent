"""
Top-level package for the Deep Research Agent project.
"""

from __future__ import annotations

from importlib import metadata as _metadata

__all__ = ["__version__"]

try:
    __version__ = _metadata.version("deep-research-agent")
except _metadata.PackageNotFoundError:  # pragma: no cover - fallback for editable installs
    __version__ = "0.0.0"
