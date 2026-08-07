"""Provenance integrity tests — smoke / benchmark identity isolation.

Verifies that every task has its own provenance (source / suite / revision)
and that smoke tasks and real benchmarks are never confused:

- GSM8K never carries smoke provenance.
- Smoke tasks are never capability_score_eligible.
- Failure paths keep the correct task identity.
- GradeResult provenance matches TaskAttempt provenance.
- Reporting correctly distinguishes eligible from ineligible tasks.
"""

import uuid
from unittest.mock import patch

import pytest

from llmtrace.adapters.lm_eval import (
    _SMOKE_MANIFEST,
    _TASK_REGISTRY,
    LmEvalAdapter,
    _get_task_def,
)
from llmtrace.benchmarks.models import (
    BenchmarkItemResult,
    GradeStatus,
    TaskAttempt,
    TaskSpec,
    TaskStatus,
)
from llmtrace.reporting.benchmark_mapper import _is_smoke_task_from_metadata
from llmtrace.scoring.aggregator import TaskScoringRegistry
from llmtrace.scoring.models import CapabilityDimension, TaskScoringSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SMOKE_DEF = _TASK_REGISTRY["llmtrace_smoke"]
_GSM8K_DEF = _TASK_REGISTRY["gsm8k_subset"]

_MATH_CLASSIFIER = TaskScoringRegistry(
    specs=[
        TaskScoringSpec(
            task_id="gsm8k_subset",
            dimension=CapabilityDimension.MATH_SCIENCE,
            task_weight=1.0,
            capability_score_eligible=True,
        ),
        TaskScoringSpec(
            task_id="llmtrace_smoke",
            dimension=CapabilityDimension.MATH_SCIENCE,
            task_weight=1.0,
            capability_score_eligible=False,
        ),
    ],
)


# ---------------------------------------------------------------------------
# 1. Smoke identity
# ---------------------------------------------------------------------------


class TestSmokeIdentity:
    """Verify that the smoke task definition carries correct smoke identity."""

    def test_smoke_is_registered(self) -> None:
        assert "llmtrace_smoke" in _TASK_REGISTRY

    def test_smoke_source_id(self) -> None:
        assert _SMOKE_DEF.source_id == "lm-eval"

    def test_smoke_is_smoke(self) -> None:
        assert _SMOKE_DEF.is_smoke is True

    def test_smoke_capability_score_eligible_false(self) -> None:
        assert _SMOKE_DEF.capability_score_eligible is False

    def test_smoke_provenance_dict(self) -> None:
        prov = _SMOKE_DEF.provenance_dict()
        assert prov["source_id"] == "lm-eval"
        assert prov["suite_id"] == "llmtrace_smoke"

    def test_smoke_metadata_has_smoke_flag(self) -> None:
        meta = _SMOKE_DEF.task_metadata()
        assert meta.get("llmtrace_smoke_task") is True

    def test_legacy_manifest_converts_to_definition(self) -> None:
        defn = _SMOKE_MANIFEST.to_definition()
        assert defn.task_id == "llmtrace_smoke"
        assert defn.is_smoke is True
        assert defn.capability_score_eligible is False


# ---------------------------------------------------------------------------
# 2. GSM8K identity
# ---------------------------------------------------------------------------


class TestGsm8kIdentity:
    """Verify that the GSM8K task definition carries correct benchmark identity."""

    def test_gsm8k_is_registered(self) -> None:
        assert "gsm8k_subset" in _TASK_REGISTRY

    def test_gsm8k_source_id(self) -> None:
        assert _GSM8K_DEF.source_id == "gsm8k"

    def test_gsm8k_is_not_smoke(self) -> None:
        assert _GSM8K_DEF.is_smoke is False

    def test_gsm8k_capability_score_eligible_true(self) -> None:
        assert _GSM8K_DEF.capability_score_eligible is True

    def test_gsm8k_provenance_dict(self) -> None:
        prov = _GSM8K_DEF.provenance_dict()
        assert prov["source_id"] == "gsm8k"
        assert prov["suite_id"] == "llmtrace-v0.2-acceptance"

    def test_gsm8k_metadata_has_no_smoke_flag(self) -> None:
        meta = _GSM8K_DEF.task_metadata()
        assert "llmtrace_smoke_task" not in meta

    def test_gsm8k_metadata_has_upstream_info(self) -> None:
        meta = _GSM8K_DEF.task_metadata()
        assert meta.get("benchmark_source") == "openai/gsm8k"
        assert meta.get("upstream_task") == "gsm8k"

    def test_gsm8k_source_revision_not_drifting(self) -> None:
        """Source revision must not be 'main' (drifting)."""
        assert _GSM8K_DEF.source_revision != "main"
        assert len(_GSM8K_DEF.source_revision) > 0


# ---------------------------------------------------------------------------
# 3. TaskAttempt provenance (success path)
# ---------------------------------------------------------------------------


class TestTaskAttemptProvenanceSuccess:
    """Verify TaskAttempt provenance matches the task definition."""

    @pytest.mark.asyncio
    async def test_gsm8k_attempt_has_gsm8k_provenance(self) -> None:
        """GSM8K TaskAttempt must carry GSM8K provenance, not smoke."""
        adapter = LmEvalAdapter()

        from tests.adapters.conftest import FakeProvider

        provider = FakeProvider()

        task = TaskSpec(task_id="gsm8k_subset", name="GSM8K", num_samples=8)
        attempt = await adapter.run_task(task, provider)

        assert attempt.source_id == "gsm8k"
        assert attempt.suite_id == "llmtrace-v0.2-acceptance"
        assert attempt.task_id == "gsm8k_subset"

    @pytest.mark.asyncio
    async def test_gsm8k_attempt_not_smoke(self) -> None:
        """GSM8K attempt metadata must NOT have llmtrace_smoke_task=True."""
        adapter = LmEvalAdapter()

        from tests.adapters.conftest import FakeProvider

        provider = FakeProvider()

        task = TaskSpec(task_id="gsm8k_subset", name="GSM8K", num_samples=8)
        attempt = await adapter.run_task(task, provider)

        assert attempt.metadata.get("llmtrace_smoke_task") is not True

    @pytest.mark.asyncio
    async def test_smoke_attempt_has_smoke_provenance(self) -> None:
        """Smoke TaskAttempt must carry smoke provenance."""
        adapter = LmEvalAdapter()

        from tests.adapters.conftest import FakeProvider

        provider = FakeProvider()

        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
        attempt = await adapter.run_task(task, provider)

        assert attempt.source_id == "lm-eval"
        assert attempt.suite_id == "llmtrace_smoke"
        assert attempt.metadata.get("llmtrace_smoke_task") is True


# ---------------------------------------------------------------------------
# 4. TaskAttempt provenance (failure paths)
# ---------------------------------------------------------------------------


class TestTaskAttemptProvenanceFailure:
    """Verify failure paths do not revert to smoke provenance."""

    @pytest.mark.asyncio
    async def test_gsm8k_failure_still_gsm8k_provenance(self) -> None:
        """GSM8K failure (invalid result) must still carry GSM8K provenance."""
        adapter = LmEvalAdapter()

        from tests.adapters.conftest import FakeProvider

        provider = FakeProvider()

        task = TaskSpec(task_id="gsm8k_subset", name="GSM8K", num_samples=1)

        # Patch runner to return a result with options_inconsistent
        with patch(
            "llmtrace.adapters.lm_eval.LmEvalRunner.run_task",
            return_value={
                "task_name": "gsm8k_subset",
                "options_inconsistent": True,
                "evidence_ids": [],
            },
        ):
            attempt = await adapter.run_task(task, provider)

        assert attempt.status == TaskStatus.FAILURE
        assert attempt.source_id == "gsm8k", "FAILURE path must keep GSM8K provenance"
        assert attempt.suite_id == "llmtrace-v0.2-acceptance"
        assert attempt.metadata.get("llmtrace_smoke_task") is not True


# ---------------------------------------------------------------------------
# 5. GradeResult provenance
# ---------------------------------------------------------------------------


class TestGradeResultProvenance:
    """Verify GradeResult provenance matches the corresponding task definition."""

    def test_gsm8k_grade_uses_gsm8k_provenance(self) -> None:
        """normalize_result for GSM8K must produce GSM8K provenance."""
        adapter = LmEvalAdapter()
        items = [
            BenchmarkItemResult(
                item_id=f"item-{i:03d}",
                task_id="gsm8k_subset",
                attempt_id="attempt-gsm",
                raw_score=0.5,
                normalized_score=0.5,
            )
            for i in range(8)
        ]
        grade = adapter.normalize_result(
            {
                "results": {"exact_match": 0.5},
                "evidence_ids": [str(uuid.uuid4())],
                "task_name": "gsm8k_subset",
                "attempt_id": "attempt-gsm",
                "item_results": [it.model_dump() for it in items],
                "planned_item_count": 8,
            }
        )
        assert grade.source_id == "gsm8k"
        assert grade.suite_id == "llmtrace-v0.2-acceptance"

    def test_gsm8k_grade_not_smoke(self) -> None:
        adapter = LmEvalAdapter()
        items = [
            BenchmarkItemResult(
                item_id=f"item-{i:03d}",
                task_id="gsm8k_subset",
                attempt_id="attempt-gsm",
                raw_score=0.5,
                normalized_score=0.5,
            )
            for i in range(8)
        ]
        grade = adapter.normalize_result(
            {
                "results": {"exact_match": 0.5},
                "evidence_ids": [],
                "task_name": "gsm8k_subset",
                "attempt_id": "attempt-gsm",
                "item_results": [it.model_dump() for it in items],
                "planned_item_count": 8,
            }
        )
        assert grade.source_id != "lm-eval", "GSM8K grade must not have smoke source"
        assert grade.status == GradeStatus.GRADED

    def test_smoke_grade_uses_smoke_provenance(self) -> None:
        """normalize_result for smoke must produce smoke provenance."""
        adapter = LmEvalAdapter()
        grade = adapter.normalize_result(
            {
                "results": {"exact_match": 1.0},
                "evidence_ids": [],
                "task_name": "llmtrace_smoke",
                "attempt_id": "attempt-smoke",
            }
        )
        assert grade.source_id == "lm-eval"
        assert grade.suite_id == "llmtrace_smoke"

    def test_gsm8k_ungradable_still_gsm8k_provenance(self) -> None:
        """GSM8K GradeResult must carry GSM8K provenance even with edge-case inputs."""
        adapter = LmEvalAdapter()
        items = [
            BenchmarkItemResult(
                item_id=f"item-{i:03d}",
                task_id="gsm8k_subset",
                attempt_id="attempt-gsm",
                raw_score=0.0,
                normalized_score=0.0,
            )
            for i in range(8)
        ]
        grade = adapter.normalize_result(
            {
                "results": {"exact_match": 0.0},
                "evidence_ids": [],
                "task_name": "gsm8k_subset",
                "attempt_id": "attempt-gsm",
                "item_results": [it.model_dump() for it in items],
                "planned_item_count": 8,
            }
        )
        assert grade.source_id == "gsm8k"
        assert grade.suite_id == "llmtrace-v0.2-acceptance"


# ---------------------------------------------------------------------------
# 6. Reporting eligibility
# ---------------------------------------------------------------------------


class TestReportingEligibility:
    """Verify that _is_smoke_task_from_metadata correctly classifies tasks."""

    def test_gsm8k_not_detected_as_smoke(self) -> None:
        task = TaskAttempt(
            attempt_id=str(uuid.uuid4()),
            source_id="gsm8k",
            source_revision="pending-verification",
            suite_id="llmtrace-v0.2-acceptance",
            suite_version="0.1.0",
            task_id="gsm8k_subset",
            adapter_id="lm-eval",
            adapter_version="0.4.12",
            status=TaskStatus.SUCCESS,
            evidence_refs=[],
            metadata={"benchmark_source": "openai/gsm8k"},
        )
        assert _is_smoke_task_from_metadata(task) is False

    def test_smoke_detected_as_smoke(self) -> None:
        task = TaskAttempt(
            attempt_id=str(uuid.uuid4()),
            source_id="lm-eval",
            source_revision="0000000-smoke",
            suite_id="llmtrace_smoke",
            suite_version="1.0.0",
            task_id="llmtrace_smoke",
            adapter_id="lm-eval",
            adapter_version="0.4.12",
            status=TaskStatus.SUCCESS,
            evidence_refs=[],
            metadata={"llmtrace_smoke_task": True},
        )
        assert _is_smoke_task_from_metadata(task) is True


# ---------------------------------------------------------------------------
# 7. Scoring regression
# ---------------------------------------------------------------------------


class TestScoringRegression:
    """Verify GSM8K contributions and smoke exclusions in scoring registry."""

    def test_gsm8k_registered_as_math_science(self) -> None:
        """GSM8K must be registered as math_science dimension."""
        spec = _MATH_CLASSIFIER.get("gsm8k_subset")
        assert spec is not None
        assert spec.dimension == CapabilityDimension.MATH_SCIENCE
        assert spec.capability_score_eligible is True

    def test_smoke_not_eligible(self) -> None:
        """Smoke task must NOT be capability_score_eligible."""
        spec = _MATH_CLASSIFIER.get("llmtrace_smoke")
        assert spec is not None
        assert spec.dimension == CapabilityDimension.MATH_SCIENCE
        assert spec.capability_score_eligible is False

    def test_unknown_task_not_registered(self) -> None:
        """Unknown tasks must not be in the registry."""
        spec = _MATH_CLASSIFIER.get("nonexistent-task")
        assert spec is None


# ---------------------------------------------------------------------------
# 8. _get_task_def fallback
# ---------------------------------------------------------------------------


class TestGetTaskDef:
    """Verify _get_task_def provides safe fallback for unknown tasks."""

    def test_known_task_returns_correct_definition(self) -> None:
        defn = _get_task_def("gsm8k_subset")
        assert defn.task_id == "gsm8k_subset"
        assert defn.is_smoke is False

    def test_unknown_task_falls_back_to_smoke(self) -> None:
        """Unknown task_id gets smoke provenance as safe default."""
        defn = _get_task_def("unknown-task-xyz")
        assert defn.task_id == "llmtrace_smoke"
        assert defn.is_smoke is True  # safe default for unknown
