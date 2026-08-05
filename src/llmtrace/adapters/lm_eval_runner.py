"""lm-eval runner — isolation boundary for lm-evaluation-harness.

The Runner encapsulates lm-eval's TaskManager lifecycle so that the
LmEvalAdapter never directly calls lm-eval internals.  This boundary
can later be replaced with a subprocess executor for stronger isolation.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from llmtrace.adapters.lm_eval_bridge import ProviderBackedLM

try:
    import lm_eval  # noqa: F401
except ImportError:
    lm_eval = None


class LmEvalNotInstalledError(RuntimeError):
    """Raised when lm-evaluation-harness is not installed."""


def _require_lm_eval() -> None:
    if lm_eval is None:
        raise LmEvalNotInstalledError(
            "lm-evaluation-harness is not installed. Install it with: pip install -e '.[lm-eval]'"
        )


class LmEvalRunner:
    """Isolated runner for lm-eval tasks via the Provider-backed LM bridge.

    Security constraints enforced here:
    - No trust_remote_code
    - No confirm_run_unsafe_code
    - No automatic task downloads
    - No dynamic plugin discovery
    - No execution of user-supplied YAML/Python code outside include_path
    - No real API calls (Provider must be injected)
    """

    def __init__(
        self,
        provider: object,
        model_name: str,
        *,
        include_path: str,
        generation_kwargs: dict[str, object] | None = None,
    ) -> None:
        _require_lm_eval()
        self._provider = provider
        self._model_name = model_name
        self._include_path = include_path
        self._generation_kwargs: dict[str, object] = generation_kwargs or {}

        self._lm: ProviderBackedLM | None = None
        self._evidence_registry: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _chdir(self) -> Iterator[None]:
        """Temporarily change to the include_path so relative data_files resolve."""
        cwd = Path.cwd()
        try:
            os.chdir(self._include_path)
            yield
        finally:
            os.chdir(cwd)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_task(
        self,
        task_name: str,
        num_fewshot: int = 0,
        batch_size: int = 1,
    ) -> dict[str, object]:
        """Execute a single lm-eval task and return controlled results.

        Args:
            task_name: lm-eval task name (must exist in include_path).
            num_fewshot: Number of few-shot examples.
            batch_size: Batch size (default 1 for deterministic execution).

        Returns:
            Controlled result dict containing at minimum:
            - results: per-metric scores
            - version: lm-eval version string
            - task_name: task identifier
            - evidence_ids: list of generated evidence UUIDs
        """
        _require_lm_eval()

        # Build the LM bridge
        self._evidence_registry.clear()
        self._lm = ProviderBackedLM(
            provider=self._provider,
            model_name=self._model_name,
            evidence_registry=self._evidence_registry,
            generation_kwargs=self._generation_kwargs,
        )

        # Use lm-eval's simple_evaluate with a TaskManager for local tasks
        from lm_eval import simple_evaluate
        from lm_eval.tasks import TaskManager

        manager = TaskManager(
            include_path=self._include_path,
            include_defaults=False,
        )

        results = None
        with self._chdir():
            results = simple_evaluate(
                model=self._lm,
                tasks=[task_name],
                num_fewshot=num_fewshot,
                batch_size=str(batch_size),
                task_manager=manager,
                confirm_run_unsafe_code=False,
                log_samples=False,
                predict_only=False,
                random_seed=1234,
                numpy_random_seed=1234,
                torch_random_seed=1234,
                fewshot_random_seed=1234,
            )

        # Build controlled result dict
        evidence_ids = list(self._evidence_registry.keys())

        # Extract per-task results from the EvalResults object
        # simple_evaluate returns a dict with keys: results, group_subtasks, configs, versions, n-shot
        task_results: dict[str, object] = {}
        if results is not None and isinstance(results, dict):
            raw_results: object = results.get("results", {})
            if isinstance(raw_results, dict):
                for _task_name, metrics in raw_results.items():
                    if isinstance(metrics, dict):
                        task_results[str(_task_name)] = dict(metrics)

        import lm_eval as pkg  # noqa: F811

        return {
            "results": task_results,
            "version": getattr(pkg, "__version__", "unknown"),
            "evidence_ids": evidence_ids,
            "task_name": task_name,
        }

    @property
    def evidence_registry(self) -> dict[str, Any]:
        """Return the evidence registry collected during the last run."""
        return dict(self._evidence_registry)
