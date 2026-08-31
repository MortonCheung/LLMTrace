"""Quick Suite runner — executes the fixed 32-item suite as a service."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, Field

from llmtrace.adapters.code_execution import CodeExecutionBackend
from llmtrace.adapters.quick_suite import (
    QuickSuiteAdapter,
)
from llmtrace.benchmarks.models import BenchmarkRunResult, RunPlan
from llmtrace.reporting.benchmark_mapper import build_benchmark_report_section
from llmtrace.reporting.benchmark_models import BenchmarkReportSection

if TYPE_CHECKING:
    from llmtrace.providers.base import BaseProvider


class QuickSuiteExecutionResult(BaseModel):
    """Canonical artefacts of one full Quick Suite execution."""

    plans: tuple[RunPlan, ...] = Field(default_factory=tuple, description="One RunPlan per task")
    run_results: tuple[BenchmarkRunResult, ...] = Field(
        default_factory=tuple, description="One BenchmarkRunResult per task (real per-task provenance)"
    )
    report_sections: tuple[BenchmarkReportSection, ...] = Field(default_factory=tuple)

    model_config = {"frozen": True, "extra": "forbid"}


class QuickSuiteRunner:
    """Run all four Quick Suite tasks against the live provider.

    Provenance stays per-task (ARC / HumanEval / GSM8K / IFEval each keep
    their own ``source_id`` / ``source_revision``) — the four sources are
    never merged into a fake single source.
    """

    def __init__(
        self,
        provider: BaseProvider,
        *,
        code_backend: CodeExecutionBackend,
    ) -> None:
        self._provider = provider
        self._adapter = QuickSuiteAdapter(code_backend=code_backend)

    async def run(self) -> QuickSuiteExecutionResult:
        plans: list[RunPlan] = []
        runs: list[BenchmarkRunResult] = []
        sections: list[BenchmarkReportSection] = []

        task_specs = self._adapter.list_tasks()

        for spec in task_specs:
            task_def = self._adapter.get_task_definition(spec.task_id)
            plan = self._adapter.build_plan(
                task_def.suite_id,
                task_def.suite_version,
                task_def.source_id,
                task_def.source_revision,
                [spec.task_id],
            )
            plans.append(plan)

            attempt = await self._adapter.run_task(spec, self._provider)

            raw_result = {
                "results": {},
                "evidence_ids": list(attempt.evidence_refs),
                "task_name": spec.task_id,
                "attempt_id": attempt.attempt_id,
                "item_results": [it.model_dump() for it in attempt.item_results],
                "planned_item_count": spec.num_samples,
            }
            grade = self._adapter.normalize_result(raw_result)

            run = BenchmarkRunResult(
                run_id=str(uuid4()),
                task_attempts=[attempt],
                grade_results=[grade],
                started_at=attempt.started_at or datetime.now(UTC),
                finished_at=attempt.finished_at or datetime.now(UTC),
                source_id=task_def.source_id,
                source_revision=task_def.source_revision,
                suite_id=task_def.suite_id,
                suite_version=task_def.suite_version,
                adapter_id=self._adapter.adapter_id,
                adapter_version=self._adapter.adapter_version,
            )
            runs.append(run)

            section = build_benchmark_report_section(plan, run)
            sections.append(section)

        return QuickSuiteExecutionResult(
            plans=tuple(plans),
            run_results=tuple(runs),
            report_sections=tuple(sections),
        )
