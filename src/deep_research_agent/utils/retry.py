from __future__ import annotations
from collections.abc import Callable
from typing import Any, TypeVar
from tenacity import (
    RetryCallState,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from deep_research_agent.utils import get_logger
T = TypeVar("T")

class RetryPolicy:
    """Standard retry policy for HTTP/API operations."""

    def __init__(
        self,
        *,
        attempts: int = 3,
        base: float = 1.0,
        max_wait: float = 8.0,
        retry_exceptions: tuple[type[Exception], ...] = (
            TimeoutError,
            ConnectionError,
            OSError,
        ),
    ) -> None:
        self.attempts = attempts
        self.base = base
        self.max_wait = max_wait
        self.retry_exceptions = retry_exceptions
        self.logger = get_logger(__name__)

    def _before_sleep(self, retry_state: RetryCallState) -> None:
        self.logger.warning(
            "retry.backoff",
            attempt=retry_state.attempt_number,
            wait=retry_state.next_action.sleep if retry_state.next_action else None,
            last_exception=str(retry_state.outcome.exception()) if retry_state.outcome else None,
        )

    def wrap(self, func: Callable[..., T]) -> Callable[..., T]:
        """Return a wrapped function with retry logic applied."""

        def wrapped(*args: Any, **kwargs: Any) -> T:
            retrying = Retrying(
                stop=stop_after_attempt(self.attempts),
                wait=wait_exponential_jitter(initial=self.base, max=self.max_wait),
                retry=retry_if_exception_type(self.retry_exceptions),
                before_sleep=self._before_sleep,
                reraise=True,
            )
            try:
                for attempt in retrying:
                    with attempt:
                        return func(*args, **kwargs)
            except RetryError as exc:
                raise exc.last_attempt.result() from exc

        return wrapped
