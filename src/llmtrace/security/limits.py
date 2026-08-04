"""请求限制控制."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RateLimit:
    """请求速率限制."""

    max_retries: int = 2
    retry_delay: float = 2.0
    backoff_factor: float = 2.0
    attempt_log: list[dict[str, Any]] = field(default_factory=list)

    async def execute_with_retry(
        self,
        coro_factory: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """带重试的异步执行."""
        last_exception: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                result = await coro_factory(*args, **kwargs)
                if attempt > 0:
                    self.attempt_log.append({"attempt": attempt + 1, "status": "success"})
                return result
            except Exception as e:
                last_exception = e
                self.attempt_log.append({"attempt": attempt + 1, "status": "error", "error": str(e)})
                if attempt < self.max_retries:
                    delay = self.retry_delay * (self.backoff_factor**attempt)
                    await asyncio.sleep(delay)

        raise last_exception  # type: ignore[misc]
