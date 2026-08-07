"""Conftest for integration tests — shares lm-eval fixtures."""

from __future__ import annotations

# Import shared fixtures from the adapters conftest
from tests.adapters.conftest import (  # noqa: F401
    FakeProvider,
    FakeProviderError,
    empty_response_provider,
    exception_evidence_provider,
    failing_provider,
    http_401_provider,
    http_429_provider,
    http_500_provider,
    smoke_provider,
    smoke_task_path,
)
