"""lm-eval runner — isolation boundary for lm-evaluation-harness.

The Runner encapsulates lm-eval's TaskManager lifecycle so that the
LmEvalAdapter never directly calls lm-eval internals.  This boundary
can later be replaced with a subprocess executor for stronger isolation.

Security invariants enforced here:
- Only whitelisted task names allowed
- Only generate_until output_type allowed
- Only local JSON dataset_path allowed
- data_files must resolve within the trusted task root
- No !function tags allowed in YAML
- No os.chdir() — absolute paths only
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

from llmtrace.adapters.lm_eval_bridge import (
    LmEvalOptionsInconsistentError,
    ProviderBackedLM,
)
from llmtrace.benchmarks.models import CompletionOptions, CompletionProvider

try:
    import lm_eval  # noqa: F401
except ImportError:
    lm_eval = None

# ---------------------------------------------------------------------------
# Security constants
# ---------------------------------------------------------------------------

# Only these task names are allowed
_TRUSTED_TASK_WHITELIST: frozenset[str] = frozenset({"llmtrace_smoke"})

# Only generate_until is allowed
_TRUSTED_OUTPUT_TYPES: frozenset[str] = frozenset({"generate_until"})

# Allowed dataset_path values (local JSON only)
_TRUSTED_DATASET_PATHS: frozenset[str] = frozenset({"json"})

# Forbidden YAML constructs
_FORBIDDEN_YAML_PATTERN = re.compile(r"!\w+")

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LmEvalNotInstalledError(RuntimeError):
    """Raised when lm-evaluation-harness is not installed."""


class LmEvalSecurityError(RuntimeError):
    """Raised when a task or path violates the Runner's security policy."""


class LmEvalValidationError(RuntimeError):
    """Raised when task configuration fails pre-execution validation."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_lm_eval() -> None:
    if lm_eval is None:
        raise LmEvalNotInstalledError(
            "lm-evaluation-harness is not installed. Install it with: pip install -e '.[lm-eval]'"
        )


def _validate_task_name(task_name: str) -> None:
    if task_name not in _TRUSTED_TASK_WHITELIST:
        raise LmEvalSecurityError(
            f"Task '{task_name}' is not in the trusted whitelist. Allowed tasks: {sorted(_TRUSTED_TASK_WHITELIST)}"
        )


def _validate_task_yaml(task_dir: Path, task_name: str) -> None:
    """Validate the YAML task config for security compliance."""
    yaml_path = task_dir / f"{task_name}.yaml"
    if not yaml_path.exists():
        raise LmEvalValidationError(f"Task YAML not found: {yaml_path}")

    raw = yaml_path.read_text()

    # Reject !function / !join / any custom YAML tag
    if _FORBIDDEN_YAML_PATTERN.search(raw):
        raise LmEvalSecurityError(
            f"Task YAML '{yaml_path}' contains custom YAML tags (!function, !join, etc.). "
            "Custom tags are forbidden for security. Use built-in metrics and doc_to_target."
        )

    # Validate output_type
    ot_match = re.search(r"^\s*output_type\s*:\s*(\S+)", raw, re.MULTILINE)
    if ot_match:
        output_type = ot_match.group(1).strip()
        if output_type not in _TRUSTED_OUTPUT_TYPES:
            raise LmEvalSecurityError(
                f"Task '{task_name}' output_type='{output_type}' is not allowed. "
                f"Only {sorted(_TRUSTED_OUTPUT_TYPES)} are supported."
            )
    else:
        raise LmEvalValidationError(f"Task YAML '{yaml_path}' missing output_type")

    # Validate dataset_path is local JSON
    dp_match = re.search(r"^\s*dataset_path\s*:\s*(\S+)", raw, re.MULTILINE)
    if dp_match:
        dataset_path = dp_match.group(1).strip()
        if dataset_path not in _TRUSTED_DATASET_PATHS:
            raise LmEvalSecurityError(
                f"Task '{task_name}' dataset_path='{dataset_path}' is not allowed. "
                f"Only {sorted(_TRUSTED_DATASET_PATHS)} are supported (local JSON)."
            )
    else:
        raise LmEvalValidationError(f"Task YAML '{yaml_path}' missing dataset_path")

    # Validate data_files is local and resolves within the task directory
    df_match = re.search(r"^\s*data_files\s*:\s*(\S+)", raw, re.MULTILINE)
    if df_match:
        data_files = df_match.group(1).strip()
        resolved = (task_dir / data_files).resolve()
        task_dir_resolved = task_dir.resolve()
        try:
            resolved.relative_to(task_dir_resolved)
        except ValueError:
            raise LmEvalSecurityError(
                f"Task '{task_name}' data_files='{data_files}' resolves to '{resolved}' "
                f"which is outside the trusted task directory '{task_dir_resolved}'."
            ) from None
        if not resolved.exists():
            raise LmEvalValidationError(f"data_files '{data_files}' not found in {task_dir}")
    else:
        raise LmEvalValidationError(f"Task YAML '{yaml_path}' missing data_files")


def _prepare_task_dir(task_root: Path, task_name: str) -> tuple[str, str, tempfile.TemporaryDirectory[str]]:
    """Rewrite task YAML with absolute path for data_files.

    lm-eval resolves ``data_files`` relative to CWD, not include_path.
    Since we cannot use os.chdir(), we create a temporary copy of the
    YAML with an absolute ``data_files`` path inside a TemporaryDirectory.

    Returns:
        (temp_task_dir, task_name, tmpdir_context) — the caller MUST hold
        the tmpdir_context alive until lm-eval execution completes.
    """
    yaml_path = task_root / f"{task_name}.yaml"
    raw = yaml_path.read_text()

    # Replace relative data_files with absolute
    df_match = re.search(r"^(\s*data_files\s*:\s*)(\S+)", raw, re.MULTILINE)
    if df_match:
        abs_data_files = (task_root / df_match.group(2)).resolve().as_posix()
        raw = raw[: df_match.start(2)] + abs_data_files + raw[df_match.end(2) :]

    # Write to a TemporaryDirectory that will auto-clean
    tmpdir = tempfile.TemporaryDirectory(prefix="llmtrace_task_")
    tmp_yaml = Path(tmpdir.name) / f"{task_name}.yaml"
    tmp_yaml.write_text(raw)

    return tmpdir.name, task_name, tmpdir


class LmEvalRunner:
    """Isolated runner for lm-eval tasks via the Provider-backed LM bridge.

    Security constraints enforced here:
    - Task name whitelist
    - Trusted task root directory
    - No arbitrary include_path from callers
    - No os.chdir() — absolute paths used throughout
    - No trust_remote_code
    - No confirm_run_unsafe_code
    - No automatic task downloads
    - No dynamic plugin discovery
    - No execution of user-supplied YAML/Python code outside task_root
    - No real API calls (Provider must be injected)
    """

    def __init__(
        self,
        provider: CompletionProvider,
        model_name: str,
        *,
        task_root: str,
        generation_kwargs: dict[str, object] | None = None,
    ) -> None:
        _require_lm_eval()
        self._provider = provider
        self._model_name = model_name
        self._task_root = Path(task_root).resolve()
        self._generation_kwargs: dict[str, object] = generation_kwargs or {}

        self._lm: ProviderBackedLM | None = None
        self._evidence_registry: dict[str, Any] = {}

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
            task_name: lm-eval task name (must be in the trusted whitelist).
            num_fewshot: Number of few-shot examples.
            batch_size: Batch size (default 1 for deterministic execution).

        Returns:
            Controlled result dict containing at minimum:
            - results: per-metric scores
            - version: lm-eval version string
            - task_name: task identifier
            - evidence_ids: list of generated evidence UUIDs
            - request_count: number of provider requests made

        Raises:
            LmEvalSecurityError: If the task or its config violate security policy.
            LmEvalValidationError: If task configuration is invalid.
            ProviderEvidenceError: If any provider request fails.
        """
        _require_lm_eval()

        # Security: validate task name and YAML before any execution
        _validate_task_name(task_name)
        _validate_task_yaml(self._task_root, task_name)

        # Prepare a temp task dir with absolute data_files path.
        # Uses TemporaryDirectory for auto-cleanup on both success and exception.
        include_path, task_name, tmpdir = _prepare_task_dir(self._task_root, task_name)

        try:
            # Build the LM bridge
            self._evidence_registry.clear()
            self._lm = ProviderBackedLM(
                provider=self._provider,
                model_name=self._model_name,
                evidence_registry=self._evidence_registry,
                generation_kwargs=self._generation_kwargs,
            )

            # Use lm-eval's simple_evaluate with a TaskManager for local tasks.
            # We use absolute paths throughout — NO os.chdir().
            from lm_eval import simple_evaluate
            from lm_eval.tasks import TaskManager

            manager = TaskManager(
                include_path=include_path,
                include_defaults=False,
            )

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
            task_results: dict[str, object] = {}
            if results is not None and isinstance(results, dict):
                raw_results: object = results.get("results", {})
                if isinstance(raw_results, dict):
                    for _tn, metrics in raw_results.items():
                        if isinstance(metrics, dict):
                            task_results[str(_tn)] = dict(metrics)

            import lm_eval as pkg  # noqa: F811

            # Check for options consistency
            options_inconsistent = False
            actual_options: CompletionOptions | None = None  # noqa: F811 — type imported above
            if self._lm is not None:
                try:
                    actual_options = self._lm.used_options
                except LmEvalOptionsInconsistentError:
                    options_inconsistent = True
                    actual_options = None

            return {
                "results": task_results,
                "version": getattr(pkg, "__version__", "unknown"),
                "evidence_ids": evidence_ids,
                "task_name": task_name,
                "request_count": len(evidence_ids),
                "actual_options": actual_options,
                "options_inconsistent": options_inconsistent,
            }
        finally:
            # Auto-clean temp directory on all paths
            tmpdir.cleanup()

    @property
    def evidence_registry(self) -> dict[str, Any]:
        """Return the evidence registry collected during the last run."""
        return dict(self._evidence_registry)
