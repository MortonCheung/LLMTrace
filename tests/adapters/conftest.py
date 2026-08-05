"""Shared fixtures for lm-eval adapter and integration tests."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from llmtrace.models.evidence import HTTPEvidence

# ---------------------------------------------------------------------------
# FakeProvider (standalone, does NOT extend BaseProvider)
# ---------------------------------------------------------------------------


class FakeProviderError(Exception):
    """Simulated provider error for testing structured failure."""


class FakeProvider:
    """A fake provider for testing the lm-eval → Provider chain.

    Returns deterministic responses based on the last message content.
    Generates real HTTPEvidence objects with UUID-backed evidence_id.
    Supports failure simulation by raising FakeProviderError.

    This is intentionally a standalone class (not inheriting from BaseProvider)
    because it must produce valid HTTPEvidence objects without real HTTP traffic.
    The ProviderBackedLM bridge calls only ``complete(model, messages)``,
    so minimal duck-typing is sufficient.
    """

    def __init__(
        self,
        response_map: dict[str, str] | None = None,
        *,
        fail_on_call: int | None = None,
        fail_error: str = "Simulated provider failure",
    ) -> None:
        """Args:
        response_map: dict of prompt substring → response text.
        fail_on_call: If set, raise error on the N-th call.
        fail_error: The error message to raise.
        """
        self.response_map = response_map or {}
        self.fail_on_call = fail_on_call
        self.fail_error = fail_error
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> HTTPEvidence:
        self.call_count += 1

        if self.fail_on_call is not None and self.call_count == self.fail_on_call:
            raise FakeProviderError(self.fail_error)

        prompt = messages[-1]["content"] if messages else ""
        response_text = self._lookup(prompt)

        self.calls.append({"model": model, "messages": messages, "response": response_text})

        # Build a valid HTTPEvidence with all required fields
        return HTTPEvidence(
            evidence_id=uuid4(),
            evidence_type="smoke_test",
            request_method="POST",
            request_url_redacted="https://fake-api.example.com/v1/chat/completions",
            request_path="/v1/chat/completions",
            request_headers_redacted={"Authorization": "Bearer sk-fake-***"},
            request_body_redacted={"model": model, "messages": [{"role": "user", "content": "[redacted]"}]},
            request_model=model,
            response_model=model,
            response_text=response_text,
            http_status=200,
            total_latency_ms=float(int(time.monotonic() * 1000) % 1000),
        )

    def _lookup(self, prompt: str) -> str:
        for key, val in self.response_map.items():
            if key in prompt:
                return val
        # Default: check for common repeat patterns
        if "LLMTRACE_OK" in prompt:
            return "LLMTRACE_OK"
        if "DETERMINISTIC" in prompt:
            return "DETERMINISTIC"
        if "ADAPTER_WORKS" in prompt:
            return "ADAPTER_WORKS"
        if "EVIDENCE_TRACED" in prompt:
            return "EVIDENCE_TRACED"
        return "LLMTRACE_OK"


# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------


@pytest.fixture
def smoke_task_path() -> str:
    return str(Path(__file__).resolve().parent.parent / "fixtures" / "lm_eval")


# ---------------------------------------------------------------------------
# Auto-use fixture: ensure lm-eval is available for integration tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _require_lm_eval_for_lm_eval_tests() -> None:
    """Skip lm-eval tests when the package is not installed."""
    try:
        import lm_eval  # noqa: F401
    except ImportError:
        pytest.skip("lm-evaluation-harness not installed")


# ---------------------------------------------------------------------------
# FakeProvider with smoke task responses
# ---------------------------------------------------------------------------


@pytest.fixture
def smoke_provider() -> FakeProvider:
    """Provider that returns the correct answer for each smoke task prompt."""
    return FakeProvider(
        response_map={
            "LLMTRACE_OK": "LLMTRACE_OK",
            "DETERMINISTIC": "DETERMINISTIC",
            "ADAPTER_WORKS": "ADAPTER_WORKS",
            "EVIDENCE_TRACED": "EVIDENCE_TRACED",
        }
    )


@pytest.fixture
def failing_provider() -> FakeProvider:
    """Provider that fails on the first call."""
    return FakeProvider(
        fail_on_call=1,
        fail_error="Simulated provider failure",
    )
