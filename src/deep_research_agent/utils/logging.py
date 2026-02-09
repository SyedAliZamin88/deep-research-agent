from __future__ import annotations
import logging
import sys
from typing import Any, Callable
import structlog
from deep_research_agent.config import settings

def _get_shared_processors() -> list[Callable[..., Any]]:
    return [
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.contextvars.merge_contextvars,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

def _configure_structlog() -> None:
    shared_processors = _get_shared_processors()

    if settings.runtime.environment == "local":
        processors: list[Callable[..., Any]] = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.rich_traceback,
            ),
        ]
    else:
        processors = [
            *shared_processors,
            structlog.processors.add_log_level,
            structlog.processors.EventRenamer("message"),
            structlog.processors.JSONRenderer(serializer=structlog.processors.JSONFallbackEncoder()),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(_python_log_level(settings.runtime.log_level)),
        cache_logger_on_first_use=True,
    )


def _configure_stdlib_logging() -> None:
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(_python_log_level(settings.runtime.log_level))


def _python_log_level(level: str) -> int:
    return getattr(logging, level.upper(), logging.INFO)


def configure_logging() -> None:
    _configure_stdlib_logging()
    _configure_structlog()


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    if not structlog.is_configured():
        configure_logging()
    return structlog.stdlib.get_logger(name)
