"""Benchmark adapter protocol.

Defines the strong-type Protocol / abstract base class that every external
benchmark harness adapter must implement.  Adapters translate external formats
into LLMTrace's unified TaskAttempt and GradeResult models, routing all HTTP
traffic through the existing Provider abstraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmtrace.benchmarks.models import BudgetEstimate, GradeResult, RunPlan, TaskAttempt, TaskSpec
    from llmtrace.providers.base import BaseProvider


class BenchmarkAdapterError(Exception):
    """Base exception for benchmark adapter errors."""


class BenchmarkAdapter(ABC):
    """Abstract base class for benchmark harness adapters.

    Design constraints:
    - Adapters MUST NOT read API keys directly.
    - Adapters MUST NOT bypass the existing Provider abstraction.
    - Adapters MUST NOT establish their own HTTP clients.
    - All outputs MUST be converted to TaskAttempt / GradeResult.
    - Failures MUST return structured error information (no swallowed exceptions).
    - The design MUST allow future subprocess isolation of external frameworks.
    """

    @property
    @abstractmethod
    def adapter_id(self) -> str:
        """Unique adapter identifier, e.g. 'lm-eval', 'livebench'."""
        ...

    @property
    @abstractmethod
    def adapter_version(self) -> str:
        """Adapter version string."""
        ...

    @abstractmethod
    def list_tasks(self) -> list[TaskSpec]:
        """List all tasks known to this adapter.

        Returns:
            A list of TaskSpec objects representing available tasks.
        """
        ...

    @abstractmethod
    def build_plan(
        self,
        suite_id: str,
        suite_version: str,
        source_id: str,
        source_revision: str,
        task_ids: list[str],
    ) -> RunPlan:
        """Build a deterministic RunPlan for the given suite and tasks.

        Args:
            suite_id: Suite identifier.
            suite_version: Suite version string.
            source_id: Benchmark source identifier.
            source_revision: Source data revision.
            task_ids: List of task IDs to include in the plan.

        Returns:
            A deterministic RunPlan.
        """
        ...

    @abstractmethod
    def estimate_budget(
        self,
        suite_id: str,
        task_ids: list[str],
        max_retries: int = 0,
    ) -> BudgetEstimate:
        """Estimate the resource budget for executing the given tasks.

        When pricing information is unavailable, estimated_cost MUST be None
        (not a fabricated value).

        Args:
            suite_id: Suite identifier.
            task_ids: List of task IDs to estimate for.
            max_retries: Maximum retry attempts per request.

        Returns:
            A BudgetEstimate with resource projections.
        """
        ...

    @abstractmethod
    async def run_task(
        self,
        task_spec: TaskSpec,
        provider: BaseProvider,
    ) -> TaskAttempt:
        """Execute a single benchmark task using the given Provider.

        Args:
            task_spec: The task specification to execute.
            provider: The Provider to use for LLM API calls.

        Returns:
            A TaskAttempt recording the execution result.
            On failure, the TaskAttempt MUST have status=FAILURE and
            provide a structured AdapterFailure in its `failure` field.
        """
        ...

    @abstractmethod
    def normalize_result(self, raw_result: dict[str, object]) -> GradeResult:
        """Normalize a raw benchmark result into a GradeResult.

        Args:
            raw_result: Raw result dictionary from the external harness.

        Returns:
            A GradeResult with normalized_score in [0, 1].

        Raises:
            BenchmarkAdapterError: If normalization fails.
        """
        ...
