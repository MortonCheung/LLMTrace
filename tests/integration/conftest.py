"""Conftest for integration tests — shares lm-eval fixtures."""

from __future__ import annotations

import pytest

# Import shared fixtures from the adapters conftest
from tests.adapters.conftest import (  # noqa: F401
    FakeProvider,
    FakeProviderError,
    failing_provider,
    smoke_provider,
    smoke_task_path,
)


@pytest.fixture(autouse=True)
def _require_lm_eval_for_integration_tests() -> None:
    """Skip lm-eval integration tests when the package is not installed."""
    try:
        import lm_eval  # noqa: F401
    except ImportError:
        pytest.skip("lm-evaluation-harness not installed")
