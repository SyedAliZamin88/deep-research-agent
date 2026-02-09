from __future__ import annotations
import asyncio
import time
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator, Iterator, Optional
from deep_research_agent.config import settings
from deep_research_agent.utils import get_logger

class RateLimiter:
    """Simple rate limiter supporting synchronous and asynchronous usage."""

    def __init__(self, max_calls: int, period_seconds: float) -> None:
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.allowance = max_calls
        self.last_check = time.monotonic()
        self.logger = get_logger(__name__)

    def _wait_if_needed(self) -> None:
        current = time.monotonic()
        time_passed = current - self.last_check
        self.last_check = current
        self.allowance += time_passed * (self.max_calls / self.period_seconds)
        if self.allowance > self.max_calls:
            self.allowance = self.max_calls

        if self.allowance < 1.0:
            wait_time = (1.0 - self.allowance) * (self.period_seconds / self.max_calls)
            self.logger.debug("ratelimiter.sleep", wait_seconds=wait_time)
            time.sleep(wait_time)
            self.allowance = 0.0
        else:
            self.allowance -= 1.0

    def acquire(self) -> None:
        self._wait_if_needed()

    @contextmanager
    def sync(self) -> Iterator[None]:
        self.acquire()
        yield


class AsyncRateLimiter:
    """Async-compatible rate limiter using asyncio primitives."""

    def __init__(self, max_calls: int, period_seconds: float) -> None:
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.allowance = max_calls
        self.last_check = time.monotonic()
        self.lock = asyncio.Lock()
        self.logger = get_logger(__name__)

    async def _wait_if_needed(self) -> None:
        async with self.lock:
            current = time.monotonic()
            time_passed = current - self.last_check
            self.last_check = current
            self.allowance += time_passed * (self.max_calls / self.period_seconds)
            if self.allowance > self.max_calls:
                self.allowance = self.max_calls

            if self.allowance < 1.0:
                wait_time = (1.0 - self.allowance) * (self.period_seconds / self.max_calls)
                self.logger.debug("async_ratelimiter.sleep", wait_seconds=wait_time)
                await asyncio.sleep(wait_time)
                self.allowance = 0.0
            else:
                self.allowance -= 1.0

    @asynccontextmanager
    async def async_context(self) -> AsyncIterator[None]:
        await self._wait_if_needed()
        yield


def default_rate_limiter() -> RateLimiter:
    return RateLimiter(
        max_calls=settings.runtime.max_concurrent_requests,
        period_seconds=1.0,
    )


def default_async_rate_limiter() -> AsyncRateLimiter:
    return AsyncRateLimiter(
        max_calls=settings.runtime.max_concurrent_requests,
        period_seconds=1.0,
    )
