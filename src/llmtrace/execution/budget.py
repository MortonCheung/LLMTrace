"""Request budget guard — hard ceiling on real target HTTP requests."""

from __future__ import annotations


class RequestBudgetExceededError(Exception):
    """Raised when a real HTTP request would exceed the execution budget."""

    error_code = "REQUEST_BUDGET_EXCEEDED"


class RequestBudget:
    """Fail-closed counter consumed once per real target HTTP request.

    ``consume()`` is called by the provider *before* sending, so a request
    that would exceed the ceiling never leaves the process.  Failed HTTP
    requests still consume budget — they are real requests the provider sent.
    """

    __slots__ = ("_maximum", "_consumed")

    def __init__(self, maximum_requests: int) -> None:
        if maximum_requests < 0:
            raise ValueError(f"maximum_requests must be >= 0, got {maximum_requests}")
        self._maximum = maximum_requests
        self._consumed = 0

    @property
    def maximum_requests(self) -> int:
        return self._maximum

    @property
    def consumed_requests(self) -> int:
        return self._consumed

    @property
    def remaining_requests(self) -> int:
        return self._maximum - self._consumed

    def consume(self, count: int = 1) -> None:
        """Reserve *count* requests; raise when the ceiling would be exceeded."""
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        if self._consumed + count > self._maximum:
            raise RequestBudgetExceededError(
                f"request budget exceeded: {self._consumed} consumed + {count} requested > {self._maximum} maximum"
            )
        self._consumed += count

    def __repr__(self) -> str:
        return f"RequestBudget(consumed={self._consumed}, maximum={self._maximum})"
