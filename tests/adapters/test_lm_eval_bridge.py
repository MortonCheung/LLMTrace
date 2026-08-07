"""Tests for ProviderBackedLM bridge."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

import pytest

pytest.importorskip("lm_eval")

from llmtrace.adapters.lm_eval_bridge import (
    ProviderBackedLM,
    ProviderEvidenceError,
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
        UUID(evidence_id)  # valid UUID

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


class TestGenerationKwargs:
    """Section 3: generation kwargs are actually passed through."""

    def test_temperature_is_passed(self, smoke_provider: object) -> None:
        """Temperature from gen_kwargs is passed as CompletionOptions."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        lm = ProviderBackedLM(
            provider=provider,
            model_name="test",
            generation_kwargs={"temperature": 0.0},
        )
        lm.generate_until([_make_instance("Repeat exactly: LLMTRACE_OK", {"temperature": 0.7})])

        assert len(provider.received_options) == 1
        opts = provider.received_options[0]
        assert opts is not None
        assert opts.temperature == 0.7

    def test_until_stop_is_passed(self, smoke_provider: object) -> None:
        """until/stop from gen_kwargs is passed as CompletionOptions."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        lm = ProviderBackedLM(
            provider=provider,
            model_name="test",
        )
        lm.generate_until([_make_instance("Repeat exactly: LLMTRACE_OK", {"until": ["\n"]})])

        assert len(provider.received_options) == 1
        opts = provider.received_options[0]
        assert opts is not None
        assert opts.until == ["\n"]

    def test_max_tokens_is_passed(self, smoke_provider: object) -> None:
        """max_gen_toks from gen_kwargs is passed as CompletionOptions."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        lm = ProviderBackedLM(
            provider=provider,
            model_name="test",
        )
        lm.generate_until([_make_instance("Repeat exactly: LLMTRACE_OK", {"max_gen_toks": 256})])

        assert len(provider.received_options) == 1
        opts = provider.received_options[0]
        assert opts is not None
        assert opts.max_gen_toks == 256

    def test_do_sample_is_passed(self, smoke_provider: object) -> None:
        """do_sample from gen_kwargs is passed as CompletionOptions."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        lm = ProviderBackedLM(
            provider=provider,
            model_name="test",
        )
        lm.generate_until([_make_instance("Repeat exactly: LLMTRACE_OK", {"do_sample": False})])

        assert len(provider.received_options) == 1
        opts = provider.received_options[0]
        assert opts is not None
        assert opts.do_sample is False

    def test_unsupported_kwargs_raise_error(self, smoke_provider: object) -> None:
        """Unsupported generation kwargs raise a structured error."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        lm = ProviderBackedLM(provider=provider, model_name="test")
        with pytest.raises(ValueError, match="Unsupported generation kwargs"):
            lm.generate_until([_make_instance("test", {"bad_param": 123})])


class TestProviderEvidenceFailure:
    """Section 4: Evidence failure checking via ProviderEvidenceError."""

    def test_exception_type_on_evidence_raises(self, exception_evidence_provider: object) -> None:
        """Evidence with exception_type raises ProviderEvidenceError."""
        from tests.adapters.conftest import FakeProvider

        provider = exception_evidence_provider
        assert isinstance(provider, FakeProvider)

        lm = ProviderBackedLM(provider=provider, model_name="test", evidence_registry={})
        with pytest.raises(ProviderEvidenceError) as exc_info:
            lm.generate_until([_make_instance("test")])

        err = exc_info.value
        assert err.error_code == "PROVIDER_EXCEPTION"
        assert err.exception_type == "ConnectionError"
        # Evidence is still saved to registry
        reg = lm._evidence_registry
        assert len(reg) == 1

    def test_http_401_raises(self, http_401_provider: object) -> None:
        """Evidence with HTTP 401 raises ProviderEvidenceError."""
        from tests.adapters.conftest import FakeProvider

        provider = http_401_provider
        assert isinstance(provider, FakeProvider)

        lm = ProviderBackedLM(provider=provider, model_name="test")
        with pytest.raises(ProviderEvidenceError) as exc_info:
            lm.generate_until([_make_instance("test")])

        err = exc_info.value
        assert err.error_code == "PROVIDER_HTTP_ERROR"
        assert err.http_status == 401
        assert err.category is not None

    def test_http_429_raises(self, http_429_provider: object) -> None:
        """Evidence with HTTP 429 raises ProviderEvidenceError."""
        from tests.adapters.conftest import FakeProvider

        provider = http_429_provider
        assert isinstance(provider, FakeProvider)

        lm = ProviderBackedLM(provider=provider, model_name="test")
        with pytest.raises(ProviderEvidenceError) as exc_info:
            lm.generate_until([_make_instance("test")])

        err = exc_info.value
        assert err.http_status == 429

    def test_http_500_raises(self, http_500_provider: object) -> None:
        """Evidence with HTTP 500 raises ProviderEvidenceError."""
        from tests.adapters.conftest import FakeProvider

        provider = http_500_provider
        assert isinstance(provider, FakeProvider)

        lm = ProviderBackedLM(provider=provider, model_name="test")
        with pytest.raises(ProviderEvidenceError) as exc_info:
            lm.generate_until([_make_instance("test")])

        err = exc_info.value
        assert err.http_status == 500

    def test_empty_response_text_raises(self, empty_response_provider: object) -> None:
        """Evidence with empty response_text raises ProviderEvidenceError."""
        from tests.adapters.conftest import FakeProvider

        provider = empty_response_provider
        assert isinstance(provider, FakeProvider)

        lm = ProviderBackedLM(provider=provider, model_name="test", evidence_registry={})
        with pytest.raises(ProviderEvidenceError) as exc_info:
            lm.generate_until([_make_instance("test")])

        err = exc_info.value
        assert err.error_code == "PROVIDER_EMPTY_RESPONSE"
        # Evidence is still saved
        reg = lm._evidence_registry
        assert len(reg) == 1


class TestProviderBackedLMOptionsConsistency:
    """Section 4 fix: inconsistent options across requests must be detected."""

    def test_used_options_uniform_returns_single(self, smoke_provider: object) -> None:
        """When all requests use the same options, used_options returns the single value."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        lm = ProviderBackedLM(provider=provider, model_name="test", generation_kwargs={"temperature": 0.0})
        lm.generate_until(
            [
                _make_instance("test1", {"temperature": 0.0}),
                _make_instance("test2", {"temperature": 0.0}),
            ]
        )

        opts = lm.used_options
        assert opts is not None
        assert opts.temperature == 0.0

    def test_used_options_inconsistent_raises(self, smoke_provider: object) -> None:
        """Two requests with different temperatures raise LmEvalOptionsInconsistentError."""
        from llmtrace.adapters.lm_eval_bridge import LmEvalOptionsInconsistentError
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        lm = ProviderBackedLM(provider=provider, model_name="test")
        lm.generate_until(
            [
                _make_instance("test1", {"temperature": 0.0}),
                _make_instance("test2", {"temperature": 0.7}),
            ]
        )

        with pytest.raises(LmEvalOptionsInconsistentError, match="LM_EVAL_OPTIONS_INCONSISTENT"):
            _ = lm.used_options

    def test_used_options_empty_returns_none(self, smoke_provider: object) -> None:
        """When no requests are made, used_options returns None without raising."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        lm = ProviderBackedLM(provider=provider, model_name="test")
        assert lm.used_options is None

    def test_used_options_list_preserves_all(self, smoke_provider: object) -> None:
        """used_options_list returns every recorded option, useful for diagnostics."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        lm = ProviderBackedLM(provider=provider, model_name="test")
        lm.generate_until(
            [
                _make_instance("test1", {"temperature": 0.0}),
                _make_instance("test2", {"temperature": 0.7}),
            ]
        )

        assert len(lm.used_options_list) == 2
        assert lm.used_options_list[0] is not None
        assert lm.used_options_list[1] is not None


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
