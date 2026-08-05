"""Shared fixtures for lm-eval adapter and integration tests."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from llmtrace.benchmarks.models import CompletionOptions
from llmtrace.models.evidence import HTTPEvidence

# ---------------------------------------------------------------------------
# FakeProvider (standalone, does NOT extend BaseProvider)
# ---------------------------------------------------------------------------


class FakeProviderError(Exception):
    """Simulated provider error for testing structured failure."""


class FakeProvider:
    """A fake provider for testing the lm-eval -> Provider chain.

    Returns deterministic responses based on the last message content.
    Generates real HTTPEvidence objects with UUID-backed evidence_id.
    Supports multiple failure simulation modes and records received options.

    This is intentionally a standalone class (not inheriting from BaseProvider)
    because it must produce valid HTTPEvidence objects without real HTTP traffic.
    The ProviderBackedLM bridge calls ``complete(model, messages, options=...)``,
    so minimal duck-typing is sufficient.
    """

    def __init__(
        self,
        response_map: dict[str, str] | None = None,
        *,
        fail_on_call: int | None = None,
        fail_error: str = "Simulated provider failure",
        # Failure mode: return evidence with exception_type set
        fail_with_exception_type: str | None = None,
        fail_with_exception_message: str | None = None,
        # Failure mode: return specific HTTP status
        fail_with_http_status: int | None = None,
        # Failure mode: return empty response_text
        fail_with_empty_response: bool = False,
    ) -> None:
        """Args:
        response_map: dict of prompt substring -> response text.
        fail_on_call: If set, raise error on the N-th call.
        fail_error: The error message to raise.
        fail_with_exception_type: Set exception_type on the returned evidence.
        fail_with_exception_message: Set exception_message on the returned evidence.
        fail_with_http_status: Override the evidence http_status to this value.
        fail_with_empty_response: Return evidence with empty response_text.
        """
        self.response_map = response_map or {}
        self.fail_on_call = fail_on_call
        self.fail_error = fail_error
        self.fail_with_exception_type = fail_with_exception_type
        self.fail_with_exception_message = fail_with_exception_message
        self.fail_with_http_status = fail_with_http_status
        self.fail_with_empty_response = fail_with_empty_response
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []
        self.received_options: list[CompletionOptions | None] = []

    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        options: CompletionOptions | None = None,
    ) -> HTTPEvidence:
        self.call_count += 1
        self.received_options.append(options)

        if self.fail_on_call is not None and self.call_count == self.fail_on_call:
            raise FakeProviderError(self.fail_error)

        prompt = messages[-1]["content"] if messages else ""
        response_text = self._lookup(prompt)

        self.calls.append({"model": model, "messages": messages, "response": response_text, "options": options})

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
            response_text="" if self.fail_with_empty_response else response_text,
            http_status=self.fail_with_http_status if self.fail_with_http_status is not None else 200,
            total_latency_ms=float(int(time.monotonic() * 1000) % 1000),
            exception_type=self.fail_with_exception_type,
            exception_message=self.fail_with_exception_message,
        )

    def _lookup(self, prompt: str) -> str:
        for key, val in self.response_map.items():
            if key in prompt:
                return val
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
    """Provider that fails on the first call by raising an exception."""
    return FakeProvider(
        fail_on_call=1,
        fail_error="Simulated provider failure",
    )


@pytest.fixture
def exception_evidence_provider() -> FakeProvider:
    """Provider that returns evidence with an exception_type set."""
    return FakeProvider(
        fail_with_exception_type="ConnectionError",
        fail_with_exception_message="Simulated connection error",
    )


@pytest.fixture
def http_401_provider() -> FakeProvider:
    """Provider that returns evidence with HTTP 401 status."""
    return FakeProvider(fail_with_http_status=401)


@pytest.fixture
def http_429_provider() -> FakeProvider:
    """Provider that returns evidence with HTTP 429 status."""
    return FakeProvider(fail_with_http_status=429)


@pytest.fixture
def http_500_provider() -> FakeProvider:
    """Provider that returns evidence with HTTP 500 status."""
    return FakeProvider(fail_with_http_status=500)


@pytest.fixture
def empty_response_provider() -> FakeProvider:
    """Provider that returns evidence with empty response_text."""
    return FakeProvider(fail_with_empty_response=True)
