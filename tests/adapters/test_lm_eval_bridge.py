"""Tests for ProviderBackedLM bridge."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llmtrace.adapters.lm_eval_bridge import (
    ProviderBackedLM,
    UnsupportedRequestTypeError,
)


def _make_instance(prompt: str, gen_kwargs: dict[str, object] | None = None) -> SimpleNamespace:
    """Create a mock lm-eval Instance for testing."""
    return SimpleNamespace(args=(prompt, gen_kwargs or {}))


class TestProviderBackedLMGenerateUntil:
    def test_generate_until_returns_responses(self, smoke_provider: object) -> None:
        """generate_until returns one response per request."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        lm = ProviderBackedLM(provider=provider, model_name="test")
        results = lm.generate_until(
            [
                _make_instance("Repeat exactly: LLMTRACE_OK"),
                _make_instance("Repeat exactly: DETERMINISTIC"),
            ]
        )
        assert len(results) == 2
        assert results[0] == "LLMTRACE_OK"
        assert results[1] == "DETERMINISTIC"

    def test_generate_until_produces_evidence(self, smoke_provider: object) -> None:
        """Each generate_until call stores evidence in registry."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        registry: dict[str, object] = {}
        lm = ProviderBackedLM(
            provider=provider,
            model_name="test",
            evidence_registry=registry,
        )
        lm.generate_until([_make_instance("Repeat exactly: ADAPTER_WORKS")])

        assert len(registry) == 1
        evidence_id = next(iter(registry.keys()))
        from uuid import UUID

        UUID(evidence_id)  # valid UUID

    def test_generate_until_passes_generation_kwargs(self, smoke_provider: object) -> None:
        """Generation kwargs are merged into prompt kwargs."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        lm = ProviderBackedLM(
            provider=provider,
            model_name="test",
            generation_kwargs={"temperature": 0.0},
        )
        results = lm.generate_until([_make_instance("Repeat exactly: LLMTRACE_OK", {"temperature": 0.0})])
        assert results[0] == "LLMTRACE_OK"

    def test_provider_call_count_matches_requests(self, smoke_provider: object) -> None:
        """Number of provider calls equals number of requests."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        lm = ProviderBackedLM(provider=provider, model_name="test")
        lm.generate_until(
            [
                _make_instance("Repeat exactly: LLMTRACE_OK"),
                _make_instance("Repeat exactly: DETERMINISTIC"),
                _make_instance("Repeat exactly: ADAPTER_WORKS"),
                _make_instance("Repeat exactly: EVIDENCE_TRACED"),
            ]
        )
        assert provider.call_count == 4


class TestProviderBackedLMUnsupported:
    def test_loglikelihood_raises_error(self, smoke_provider: object) -> None:
        """loglikelihood raises UnsupportedRequestTypeError."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        lm = ProviderBackedLM(provider=provider, model_name="test")
        with pytest.raises(
            UnsupportedRequestTypeError,
            match="loglikelihood.*not supported",
        ):
            lm.loglikelihood([])

    def test_loglikelihood_rolling_raises_error(self, smoke_provider: object) -> None:
        """loglikelihood_rolling raises UnsupportedRequestTypeError."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        lm = ProviderBackedLM(provider=provider, model_name="test")
        with pytest.raises(
            UnsupportedRequestTypeError,
            match="loglikelihood_rolling.*not supported",
        ):
            lm.loglikelihood_rolling([])
