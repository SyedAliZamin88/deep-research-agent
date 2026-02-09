from __future__ import annotations
import asyncio
from typing import Awaitable, TypeVar
from deep_research_agent.core.interfaces import AgentNode
from deep_research_agent.utils import get_logger
from deep_research_agent.utils.retry import RetryPolicy

T = TypeVar("T")

class BaseAgentNode(AgentNode):
    """Common functionality shared by all agent nodes."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = get_logger(f"agent.{name}")
        self.retry_policy = RetryPolicy()

    def log(self, event: str, **payload: object) -> None:
        """Emit a structured log event scoped to this node."""
        self.logger.info(event, node=self.name, **payload)

    @staticmethod
    def _await_result(coro: Awaitable[T]) -> T:
        """Synchronously obtain the result of an awaitable, handling nested loops."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            return asyncio.run_coroutine_threadsafe(coro, loop).result()

        return asyncio.run(coro)
