"""Provider-backed LM bridge for lm-evaluation-harness.

Wraps a Provider-like object so that lm-eval's synchronous generate_until
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
from typing import Any

from llmtrace.models.evidence import HTTPEvidence

# ---------------------------------------------------------------------------
# Conditional inheritance from lm-eval's LM base class
# ---------------------------------------------------------------------------

try:
    from lm_eval.api.model import LM as _LmEvalLMBase  # noqa: N811
except ImportError:  # pragma: no cover  (lm-eval is optional)
    _LmEvalLMBase = object


def _run_async_in_thread(coro: object) -> object:
    """Run an async coroutine from a synchronous context using a dedicated event loop.

    lm-eval's generate_until is synchronous, but BaseProvider.complete() is async.
    This bridge runs the coroutine in a new event loop inside a background thread,
    which avoids asyncio.run() conflicts when called from within an existing loop.
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


class UnsupportedRequestTypeError(NotImplementedError):
    """Raised when lm-eval requests a mode not yet supported by this bridge."""


class ProviderBackedLM(_LmEvalLMBase):  # type: ignore[misc]
    """An lm-eval LM that delegates generate_until to a Provider-like object.

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
        provider: Any,
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

    # ------------------------------------------------------------------
    # generate_until — the only supported mode
    # ------------------------------------------------------------------

    def generate_until(self, requests: list[object]) -> list[str]:
        """Generate completions for each Instance request.

        Each Instance for generate_until has arguments = (ctx, gen_kwargs).
        Each completion is obtained via Provider.complete(), producing
        a real HTTPEvidence object that is stored in evidence_registry.

        Returns:
            List of generated text strings, one per request.
        """
        responses: list[str] = []
        for instance in requests:
            # Extract (prompt, gen_kwargs) from Instance.arguments
            # We use duck-typing to avoid importing lm_eval.api.instance.Instance
            inst_args = getattr(instance, "args", None)
            if isinstance(inst_args, tuple) and len(inst_args) >= 1:
                prompt: str = str(inst_args[0])
                gen_kwargs: dict[str, object] = (
                    inst_args[1] if len(inst_args) >= 2 and isinstance(inst_args[1], dict) else {}
                )
            else:
                prompt = str(inst_args)
                gen_kwargs = {}

            merged_kwargs: dict[str, object] = dict(self._generation_kwargs)
            merged_kwargs.update(gen_kwargs)

            # Build chat messages; lm-eval sends the formatted prompt as ctx
            messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
            evidence = _run_async_in_thread(
                self._provider.complete(self._model_name, messages),
            )
            assert isinstance(evidence, HTTPEvidence)
            self._evidence_registry[str(evidence.evidence_id)] = evidence

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
