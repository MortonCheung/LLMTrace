"""Acceptance tests: GSM8K subset through full LLMTrace vertical pipeline.

Verifies that a real upstream benchmark (GSM8K 8-sample fixed subset)
can be loaded, executed through ProviderBackedLM, scored, and reported
without calling external APIs.
"""

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from llmtrace.adapters.lm_eval import LmEvalAdapter
from llmtrace.benchmarks.models import (
    BenchmarkRunResult,
    BenchmarkSource,
    GradeResult,
    GradeStatus,
    RunPlan,
    TaskAttempt,
    TaskStatus,
)
from llmtrace.benchmarks.planner import build_plan
from llmtrace.config import AuditConfig, AuthStyle, Protocol
from llmtrace.models.audit import AuditResult, RiskLevel
from llmtrace.reporting.benchmark_mapper import build_benchmark_report_section
from llmtrace.reporting.json_report import generate_json_report
from llmtrace.scoring import (
    CapabilityDimension,
    TaskScoringRegistry,
    TaskScoringSpec,
    aggregate_capability_profile,
    aggregate_dimension_score,
)
from llmtrace.scoring.policy import CapabilityScoringPolicy
from tests.adapters.conftest import FakeProvider

# ---------------------------------------------------------------------------
# Test constants — pinned subset identity
# ---------------------------------------------------------------------------

_GSM8K_SOURCE = BenchmarkSource(
    source_id="gsm8k",
    name="GSM8K",
    description="Grade school math word problems (openai/gsm8k)",
    url="https://github.com/openai/grade-school-math",
)

_SUITE_ID = "llmtrace-v0.2-acceptance"
_SUITE_VERSION = "0.1.0"

# Each GSM8K question with its correct #### answer (raw text to feed Mock).
# These MUST match the order in gsm8k_subset.json.
_GSM8K_MOCK_RESPONSES: dict[str, str] = {
    "Janet's ducks lay 16 eggs per day": "#### 18",
    "A robe takes 2 bolts of blue fiber": "#### 3",
    "Josh decides to try flipping a house": "#### 70000",
    "James decides to run 3 sprints": "#### 540",
    "Every day, Wendi feeds each of her chickens": "#### 20",
    "Kylar went to the store to buy glasses": "#### 64",
    "Toulouse has twice as many sheep as Charleston": "#### 260",
    "Carla is downloading a 200 GB file": "#### 160",
}

_GSM8K_SCORING_REGISTRY = TaskScoringRegistry(
    [
        TaskScoringSpec(
            task_id="gsm8k_subset",
            dimension=CapabilityDimension.MATH_SCIENCE,
            task_weight=1.0,
        ),
    ]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plan(adapter, spec, run_result=None) -> RunPlan:
    """Build a RunPlan using run_result provenance (if provided) for consistency."""
    if run_result is not None:
        suite_id = run_result.suite_id
        suite_version = run_result.suite_version
        source_id = run_result.source_id
        source_revision = run_result.source_revision
    else:
        suite_id = _SUITE_ID
        suite_version = _SUITE_VERSION
        source_id = _GSM8K_SOURCE.source_id
        source_revision = "main"

    return build_plan(
        suite_id=suite_id,
        suite_version=suite_version,
        source_id=source_id,
        source_revision=source_revision,
        adapter_id=adapter.adapter_id,
        adapter_version=adapter.adapter_version,
        tasks=[spec],
    )


def _make_audit_result(evidence_ids: list | None = None) -> AuditResult:
    return AuditResult(
        config=AuditConfig(
            protocol=Protocol.OPENAI,
            base_url="https://api.example.com",
            model="test-model",
            api_key_env="TEST_KEY",
            auth_style=AuthStyle.BEARER,
            repeat_count=1,
            timeout=30.0,
            max_output_tokens=100,
            check_streaming=False,
        ),
        evidence=evidence_ids or [],
        findings=[],
        risk_level=RiskLevel.INCONCLUSIVE,
        schema_fingerprints=[],
        model_list=[],
        start_time=datetime(2026, 8, 1, tzinfo=UTC),
        end_time=datetime(2026, 8, 1, 1, tzinfo=UTC),
        llmtrace_version="0.2.0",
        python_version="3.12",
        platform="darwin",
        report_id="gsm8k-acceptance-report",
        content_hash="",
    )


def _make_run_result(attempt, grade, adapter) -> BenchmarkRunResult:
    return BenchmarkRunResult(
        run_id="00000000-0000-0000-0000-000000000555",
        task_attempts=[attempt],
        item_results=list(attempt.item_results),
        grade_results=[grade],
        evidence_refs=attempt.evidence_refs,
        source_id=attempt.source_id,
        source_revision=attempt.source_revision,
        suite_id=attempt.suite_id,
        suite_version=attempt.suite_version,
        adapter_id=adapter.adapter_id,
        adapter_version=adapter.adapter_version,
    )


async def _run_gsm8k_pipeline(responses: dict[str, str] | None = None):
    """Run the full GSM8K pipeline with the given mock responses."""
    adapter = LmEvalAdapter()
    tasks = adapter.list_tasks()
    gsm_spec = next(t for t in tasks if t.task_id == "gsm8k_subset")

    provider = FakeProvider(response_map=responses or _GSM8K_MOCK_RESPONSES)

    attempt = await adapter.run_task(gsm_spec, provider)

    raw_result = {
        "results": {},
        "evidence_ids": attempt.evidence_refs,
        "task_name": "gsm8k_subset",
        "attempt_id": attempt.attempt_id,
    }
    # Transfer the exact_match score from metadata
    metric = attempt.metadata.get("metric_result")
    raw_result["results"]["exact_match"] = metric["value"]
    grade = adapter.normalize_result(raw_result)

    run_result = _make_run_result(attempt, grade, adapter)
    plan = _make_plan(adapter, gsm_spec, run_result=run_result)

    return adapter, attempt, grade, run_result, plan, provider


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGsm8kAcceptance:
    """Full vertical pipeline: GSM8K subset -> Provider -> Evidence -> Score -> Report."""

    def test_gsm8k_subset_loads_and_plans(self) -> None:
        """Adapter can discover and plan execution for the GSM8K subset."""
        adapter = LmEvalAdapter()
        tasks = adapter.list_tasks()

        gsm_specs = [t for t in tasks if t.task_id == "gsm8k_subset"]
        assert len(gsm_specs) == 1
        spec = gsm_specs[0]
        assert spec.num_samples == 8
        assert spec.category == "benchmark"

    def test_gsm8k_subset_task_ids_deterministic(self) -> None:
        """Repeated task listing returns the same task_ids for the subset."""
        adapter = LmEvalAdapter()
        tasks1 = adapter.list_tasks()
        tasks2 = adapter.list_tasks()

        ids1 = sorted(t.task_id for t in tasks1 if t.task_id == "gsm8k_subset")
        ids2 = sorted(t.task_id for t in tasks2 if t.task_id == "gsm8k_subset")
        assert ids1 == ids2

    @pytest.mark.asyncio
    async def test_gsm8k_full_pipeline_with_mock(self) -> None:
        """Run the full vertical pipeline with a Mock Provider (no real API)."""
        adapter, attempt, grade, run_result, plan, provider = await _run_gsm8k_pipeline()

        # ---------- TaskAttempt ----------
        assert attempt.task_id == "gsm8k_subset"
        assert attempt.status == TaskStatus.SUCCESS
        assert len(attempt.evidence_refs) >= 1

        # ---------- Evidence ----------
        evidence_map = {e.evidence_id: e for e in provider.evidence}
        assert len(evidence_map) >= 1
        for eid_str in attempt.evidence_refs:
            import uuid as _uuid

            eid = _uuid.UUID(eid_str)
            assert eid in evidence_map, f"Orphan evidence_ref: {eid}"

        # ---------- Grade ----------
        assert grade.attempt_id == attempt.attempt_id
        assert grade.task_id == attempt.task_id
        assert grade.status == GradeStatus.GRADED
        assert 0.0 <= grade.normalized_score <= 1.0

        # ---------- BenchmarkRunResult ----------
        assert len(run_result.task_attempts) == 1
        assert len(run_result.grade_results) == 1

        # ---------- Capability Scoring ----------
        policy = CapabilityScoringPolicy.create_v1()
        dim_score = aggregate_dimension_score(
            CapabilityDimension.MATH_SCIENCE,
            [run_result],
            _GSM8K_SCORING_REGISTRY,
            policy,
        )
        assert dim_score.dimension == CapabilityDimension.MATH_SCIENCE
        assert dim_score.status.value == "uncalibrated"
        assert dim_score.task_coverage > 0.0
        assert dim_score.raw_normalized_score > 0.0
        assert dim_score.calibrated_score is None  # no calibration yet

        profile = aggregate_capability_profile(
            [run_result],
            _GSM8K_SCORING_REGISTRY,
            policy,
        )
        assert profile.calibrated_total_score is None

        # ---------- Report ----------
        section = build_benchmark_report_section(plan, run_result)
        assert section is not None

        audit = _make_audit_result(list(provider.evidence))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = Path(f.name)
        try:
            generate_json_report(audit, out_path, benchmark_sections=[section])
            with open(out_path) as f:
                report = json.load(f)
            assert "benchmarks" in report
        finally:
            out_path.unlink(missing_ok=True)


class TestGsm8kFailureIsolation:
    """Single task failure must not break the run or produce fake grades."""

    @pytest.mark.asyncio
    async def test_gsm8k_partial_failure_no_corruption(self) -> None:
        """Mock returns correct answers for most but wrong for one -> isolation."""
        wrong_responses = dict(_GSM8K_MOCK_RESPONSES)
        wrong_responses["Carla is downloading a 200 GB file"] = "#### 0"

        _, attempt, grade, _, _, _ = await _run_gsm8k_pipeline(wrong_responses)

        assert attempt.status == TaskStatus.SUCCESS
        # Wrong answer for one question → score = 7/8 = 0.875 (still graded)
        assert grade.status == GradeStatus.GRADED
        assert grade.normalized_score == 0.875


class TestGsm8kEvidenceClosure:
    """Evidence closure validation after full pipeline."""

    @pytest.mark.asyncio
    async def test_gsm8k_evidence_closure(self) -> None:
        """Verify that evidence_refs form a closed chain."""
        _, attempt, _, _, _, provider = await _run_gsm8k_pipeline()

        evidence_map = {e.evidence_id: e for e in provider.evidence}
        for eid_str in attempt.evidence_refs:
            import uuid as _uuid

            eid = _uuid.UUID(eid_str)
            assert eid in evidence_map, f"Orphan evidence_ref: {eid}"

    @pytest.mark.asyncio
    async def test_gsm8k_evidence_ids_unique(self) -> None:
        """All evidence_ids must be globally unique."""
        _, _, _, _, _, provider = await _run_gsm8k_pipeline()

        ids = [e.evidence_id for e in provider.evidence]
        assert len(ids) == len(set(ids)), "Duplicate evidence_id detected"


class TestGsm8kCrossRunIsolation:
    """Cross-run attempt/grade isolation for GSM8K subset."""

    def test_gsm8k_duplicate_attempt_id_across_runs_raises(self) -> None:
        """Same attempt_id across two runs -> ValueError (via scoring layer)."""
        adapter = LmEvalAdapter()

        dup_attempt = TaskAttempt(
            attempt_id="dup-att-x",
            task_id="gsm8k_subset",
            status=TaskStatus.SUCCESS,
            evidence_refs=[],
            metadata={},
            source_id=_GSM8K_SOURCE.source_id,
            source_revision="main",
            suite_id=_SUITE_ID,
            suite_version=_SUITE_VERSION,
            adapter_id=adapter.adapter_id,
            adapter_version=adapter.adapter_version,
        )

        dup_grade = GradeResult(
            grade_id="dup-grade-x",
            attempt_id="dup-att-x",
            task_id="gsm8k_subset",
            grader_id="exact_match",
            raw_score=1.0,
            status=GradeStatus.GRADED,
            normalized_score=1.0,
            evidence_refs=[],
            source_id=_GSM8K_SOURCE.source_id,
            source_revision="main",
            suite_id=_SUITE_ID,
            suite_version=_SUITE_VERSION,
            adapter_id=adapter.adapter_id,
            adapter_version=adapter.adapter_version,
        )

        run_a = BenchmarkRunResult(
            run_id="run-a",
            task_attempts=[dup_attempt],
            grade_results=[],
            evidence_refs=[],
            source_id=_GSM8K_SOURCE.source_id,
            source_revision="main",
            suite_id=_SUITE_ID,
            suite_version=_SUITE_VERSION,
            adapter_id=adapter.adapter_id,
            adapter_version=adapter.adapter_version,
        )

        run_b = BenchmarkRunResult(
            run_id="run-b",
            task_attempts=[dup_attempt],
            grade_results=[dup_grade],
            evidence_refs=[],
            source_id=_GSM8K_SOURCE.source_id,
            source_revision="main",
            suite_id=_SUITE_ID,
            suite_version=_SUITE_VERSION,
            adapter_id=adapter.adapter_id,
            adapter_version=adapter.adapter_version,
        )

        policy = CapabilityScoringPolicy.create_v1()
        with pytest.raises(ValueError, match="Duplicate attempt_id"):
            aggregate_dimension_score(
                CapabilityDimension.MATH_SCIENCE,
                [run_a, run_b],
                _GSM8K_SCORING_REGISTRY,
                policy,
            )
