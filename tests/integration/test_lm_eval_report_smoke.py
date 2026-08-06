"""Real smoke report chain test — runs in lm-eval integration CI job.

Validates the complete chain:
  LmEvalAdapter.list_tasks() → planner.build_plan()
  → LmEvalAdapter.run_task() → TaskAttempt → GradeResult
  → BenchmarkRunResult → build_benchmark_report_section()
  → TaskReportItem

Collected by CI glob: tests/integration/test_lm_eval*.py
"""

from __future__ import annotations

from uuid import uuid4

import pytest

pytest.importorskip("lm_eval")

from llmtrace.adapters.lm_eval import LmEvalAdapter
from llmtrace.benchmarks.models import (
    BenchmarkRunResult,
    GradeResult,
    GradeStatus,
)
from llmtrace.benchmarks.planner import build_plan
from llmtrace.reporting.benchmark_mapper import build_benchmark_report_section
from tests.adapters.conftest import FakeProvider


class TestRealSmokeReportChain:
    """End-to-end smoke chain: adapter → attempt → grade → report section."""

    @pytest.mark.asyncio
    async def test_real_smoke_link(self, smoke_provider: object) -> None:
        """Full smoke chain with all required assertions."""
        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        # 1. Get the smoke task spec from LmEvalAdapter
        adapter = LmEvalAdapter()
        task_specs = adapter.list_tasks()
        assert len(task_specs) == 1
        smoke_spec = task_specs[0]

        # 2. Build a plan using the shared planner
        plan = build_plan(
            suite_id="llmtrace_smoke",
            suite_version="1.0.0",
            source_id="lm-eval",
            source_revision="0000000-smoke",
            adapter_id=adapter.adapter_id,
            adapter_version=adapter.adapter_version,
            tasks=[smoke_spec],
        )

        # 3. Run the smoke task via the adapter
        attempt = await adapter.run_task(smoke_spec, provider)

        # 4. Assert llmtrace_smoke_task metadata flag
        assert attempt.metadata.get("llmtrace_smoke_task") is True

        # 5. Assert metric_result exists with generation_options
        metric_result = attempt.metadata.get("metric_result")
        assert metric_result is not None, "metadata must contain 'metric_result'"
        assert "generation_options" in metric_result, "metric_result must contain 'generation_options'"

        # 6. Assert evidence_refs non-empty
        assert len(attempt.evidence_refs) > 0

        # 7. Grade the result
        grade = GradeResult(
            grade_id=str(uuid4()),
            attempt_id=attempt.attempt_id,
            task_id=smoke_spec.task_id,
            grader_id="exact_match",
            raw_score=1.0,
            normalized_score=1.0,
            status=GradeStatus.GRADED,
            source_id="lm-eval",
            source_revision="0000000-smoke",
            suite_id="llmtrace_smoke",
            suite_version="1.0.0",
            adapter_id=adapter.adapter_id,
            adapter_version=adapter.adapter_version,
        )

        # 8. Build BenchmarkRunResult and report section
        run_result = BenchmarkRunResult(
            run_id=str(uuid4()),
            task_attempts=[attempt],
            grade_results=[grade],
            evidence_refs=attempt.evidence_refs,
            source_id="lm-eval",
            source_revision="0000000-smoke",
            suite_id="llmtrace_smoke",
            suite_version="1.0.0",
            adapter_id=adapter.adapter_id,
            adapter_version=adapter.adapter_version,
        )

        section = build_benchmark_report_section(plan, run_result)

        # 9. Assert capability_score_eligible is False
        task = section.tasks[0]
        assert task.capability_score_eligible is False

        # 10. Assert metadata preserved through the chain
        assert task.metadata.get("llmtrace_smoke_task") is True

        # 11. Assert scores match
        assert task.raw_score == 1.0
        assert task.normalized_score == 1.0

        # 12. Assert grade status is GRADED
        assert task.grade_status is not None
        assert task.grade_status.value == "graded"
