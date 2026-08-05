"""Provider-backed LM bridge for lm-evaluation-harness.

Wraps a Provider so that lm-eval's synchronous generate_until
calls are routed through LLMTrace's async Provider infrastructure.

Usage (test-only, no real API)::

    provider = FakeProvider(...)
    lm = ProviderBackedLM(
        provider=provider,
        model_name="test-model",
        evidence_registry={},
    )
    # lm-eval will call lm.generate_until(instances) internally
"""

from __future__ import annotations

import asyncio
import threading
from uuid import UUID

from llmtrace.benchmarks.models import (
    CompletionOptions,
    CompletionProvider,
    FailureCategory,
)
from llmtrace.models.evidence import HTTPEvidence

# ---------------------------------------------------------------------------
# Conditional inheritance from lm-eval's LM base class
# ---------------------------------------------------------------------------

try:
    from lm_eval.api.model import LM as _LmEvalLMBase  # noqa: N811
except ImportError:  # pragma: no cover  (lm-eval is optional)
    _LmEvalLMBase = object


# ---------------------------------------------------------------------------
# ProviderEvidenceError — structured failure from evidence inspection
# ---------------------------------------------------------------------------


class ProviderEvidenceError(Exception):
    """Raised when an Evidence indicates a failed provider request.

    ProviderBackedLM inspects every HTTPEvidence after saving it to the
    registry.  If the evidence shows a non-successful request (exception,
    non-2xx status, or empty response_text), this exception is raised
    with a structured payload that the adapter can catch and convert into
    an AdapterFailure.
    """

    def __init__(
        self,
        evidence_id: UUID,
        error_code: str,
        category: FailureCategory,
        retryable: bool,
        http_status: int | None,
        exception_type: str | None,
        message: str,
    ) -> None:
        super().__init__(message)
        self.evidence_id = evidence_id
        self.error_code = error_code
        self.category = category
        self.retryable = retryable
        self.http_status = http_status
        self.exception_type = exception_type


class UnsupportedRequestTypeError(NotImplementedError):
    """Raised when lm-eval requests a mode not yet supported by this bridge."""


# ---------------------------------------------------------------------------
# Thread-based async bridge
# ---------------------------------------------------------------------------


def _run_async_in_thread(coro: object) -> object:
    """Run an async coroutine from a synchronous context using a dedicated event loop.

    lm-eval's generate_until is synchronous, but CompletionProvider.complete()
    is async.  This bridge runs the coroutine in a new event loop inside a
    background thread, which avoids asyncio.run() conflicts when called from
    within an existing loop.
    """
    result: list[object] = []
    error: list[BaseException] = []

    def _target() -> None:
        loop = asyncio.new_event_loop()
        try:
            val: object = loop.run_until_complete(coro)  # type: ignore[arg-type]
            result.append(val)
        except BaseException as e:
            error.append(e)
        finally:
            loop.close()

    t = threading.Thread(target=_target)
    t.start()
    t.join()

    if error:
        raise error[0]
    return result[0]


# ---------------------------------------------------------------------------
# Evidence failure inspection
# ---------------------------------------------------------------------------


def _check_evidence(evidence: HTTPEvidence) -> None:
    """Inspect an HTTPEvidence and raise ProviderEvidenceError on failure.

    A request is considered failed when ANY of these hold:
    - exception_type is set (network / SDK error)
    - HTTP status is not 2xx
    - response_text is empty (and status was 200 — a suspicious "empty success")
    """
    evidence_id = evidence.evidence_id

    if evidence.exception_type:
        category = _category_from_exception(evidence.exception_type)
        raise ProviderEvidenceError(
            evidence_id=evidence_id,
            error_code="PROVIDER_EXCEPTION",
            category=category,
            retryable=category in (FailureCategory.NETWORK, FailureCategory.TIMEOUT, FailureCategory.RATE_LIMIT),
            http_status=evidence.http_status,
            exception_type=evidence.exception_type,
            message=f"Provider raised {evidence.exception_type}: {evidence.exception_message or '(no message)'}",
        )

    if evidence.http_status is not None and not (200 <= evidence.http_status < 300):
        category = _category_from_http_status(evidence.http_status)
        raise ProviderEvidenceError(
            evidence_id=evidence_id,
            error_code="PROVIDER_HTTP_ERROR",
            category=category,
            retryable=category in (FailureCategory.NETWORK, FailureCategory.TIMEOUT, FailureCategory.RATE_LIMIT),
            http_status=evidence.http_status,
            exception_type=None,
            message=f"Provider returned HTTP {evidence.http_status}",
        )

    if not evidence.response_text:
        raise ProviderEvidenceError(
            evidence_id=evidence_id,
            error_code="PROVIDER_EMPTY_RESPONSE",
            category=FailureCategory.PROVIDER,
            retryable=True,
            http_status=evidence.http_status,
            exception_type=None,
            message="Provider returned empty response_text",
        )


def _category_from_exception(exception_type: str) -> FailureCategory:
    """Map an exception class name to a FailureCategory."""
    lower = exception_type.lower()
    if "timeout" in lower:
        return FailureCategory.TIMEOUT
    if any(kw in lower for kw in ("auth", "unauthorized", "forbidden")):
        return FailureCategory.AUTH
    if any(kw in lower for kw in ("rate", "throttl", "limit")):
        return FailureCategory.RATE_LIMIT
    if any(kw in lower for kw in ("connection", "network", "dns", "resolve")):
        return FailureCategory.NETWORK
    return FailureCategory.PROVIDER


def _category_from_http_status(http_status: int) -> FailureCategory:
    """Map an HTTP status code to a FailureCategory."""
    if http_status in (401, 403):
        return FailureCategory.AUTH
    if http_status == 429:
        return FailureCategory.RATE_LIMIT
    if 500 <= http_status < 600:
        return FailureCategory.PROVIDER
    return FailureCategory.UNKNOWN


# ---------------------------------------------------------------------------
# ProviderBackedLM
# ---------------------------------------------------------------------------


class ProviderBackedLM(_LmEvalLMBase):  # type: ignore[misc]
    """An lm-eval LM that delegates generate_until to a CompletionProvider.

    Only generate_until is supported in this round.
    loglikelihood() and loglikelihood_rolling() explicitly raise
    UnsupportedRequestTypeError rather than silently returning junk.

    lm-eval passes ``list[Instance]`` to generate_until(). Each Instance
    for ``generate_until`` output_type has ``arguments = (ctx, gen_kwargs)``
    where ctx is the prompt string and gen_kwargs are the task-level
    generation_kwargs from the YAML config.

    When lm-eval is installed, this class inherits from ``lm_eval.api.model.LM``
    so that ``simple_evaluate()`` accepts it as a valid model.  When lm-eval
    is not installed, it falls back to a plain object (for testing without the
    optional dependency).
    """

    def __init__(
        self,
        provider: CompletionProvider,
        model_name: str,
        *,
        evidence_registry: dict[str, HTTPEvidence] | None = None,
        generation_kwargs: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        self._provider = provider
        self._model_name = model_name
        self._evidence_registry: dict[str, HTTPEvidence] = evidence_registry if evidence_registry is not None else {}
        self._generation_kwargs: dict[str, object] = generation_kwargs or {}
        self._used_options: list[CompletionOptions | None] = []

    @property
    def used_options(self) -> CompletionOptions | None:
        """Return the options actually passed to Provider.complete().

        If all requests used identical options, returns that single
        CompletionOptions.  If different options were used across
        requests, returns None (inconsistent options are a bug —
        the smoke task should produce uniform options per task run).
        """
        if not self._used_options:
            return None
        first = self._used_options[0]
        if all(o == first for o in self._used_options):
            return first
        return None

    # ------------------------------------------------------------------
    # generate_until — the only supported mode
    # ------------------------------------------------------------------

    def generate_until(self, requests: list[object]) -> list[str]:
        """Generate completions for each Instance request.

        Each Instance for generate_until has arguments = (ctx, gen_kwargs).
        gen_kwargs are validated through CompletionOptions.from_lm_eval_kwargs.
        After each provider call the returned HTTPEvidence is saved to the
        registry AND inspected for failure (→ ProviderEvidenceError).
        """
        responses: list[str] = []
        for instance in requests:
            # Extract (prompt, gen_kwargs) from Instance.arguments
            inst_args = getattr(instance, "args", None)
            if isinstance(inst_args, tuple) and len(inst_args) >= 1:
                prompt: str = inst_args[0]
                instance_gen_kwargs: dict[str, object] = (
                    inst_args[1] if len(inst_args) >= 2 and isinstance(inst_args[1], dict) else {}
                )
            else:
                prompt = str(inst_args)
                instance_gen_kwargs = {}

            # Merge: base generation_kwargs + per-instance overrides
            merged_kwargs: dict[str, object] = dict(self._generation_kwargs)
            merged_kwargs.update(instance_gen_kwargs)

            # Build typed CompletionOptions (fails on unsupported keys)
            options = CompletionOptions.from_lm_eval_kwargs(merged_kwargs) if merged_kwargs else None

            # Record the options actually passed to the provider
            self._used_options.append(options)

            # Build chat messages; lm-eval sends the formatted prompt as ctx
            messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]

            evidence = _run_async_in_thread(
                self._provider.complete(self._model_name, messages, options=options),
            )
            assert isinstance(evidence, HTTPEvidence)
            self._evidence_registry[str(evidence.evidence_id)] = evidence

            # Inspect evidence for failure AFTER saving (evidence is always persisted)
            _check_evidence(evidence)

            text = evidence.response_text or ""
            responses.append(text)

        return responses

    # ------------------------------------------------------------------
    # Unsupported modes — explicit errors
    # ------------------------------------------------------------------

    def loglikelihood(
        self,
        requests: list[object],
    ) -> list[tuple[float, bool]]:
        """Not supported in this round."""
        raise UnsupportedRequestTypeError(
            "loglikelihood() is not supported by ProviderBackedLM. Only generate_until() is available in this round."
        )

    def loglikelihood_rolling(
        self,
        requests: list[object],
    ) -> list[float]:
        """Not supported in this round."""
        raise UnsupportedRequestTypeError(
            "loglikelihood_rolling() is not supported by ProviderBackedLM. "
            "Only generate_until() is available in this round."
        )
