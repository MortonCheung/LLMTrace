"""Integration tests for item-level benchmark results (v0.3-A).

Tests cover production-path invariants:
- ItemAggregateResult fixed-denominator scoring
- BenchmarkItemResult status consistency validators
- TaskAttempt duplicate/cross-attempt/evidence closure validators
- GSM8K item extraction and grading (correct, wrong, ungradable, failure)
- GradeResult derived from item aggregate; lm-eval cross-check
- Provider failure isolation in bridge layer
- Reporting round-trip with item-level display
- HTML XSS escaping through real renderer
"""

from __future__ import annotations

import uuid

import pytest

from llmtrace.adapters.lm_eval import (
    LmEvalAdapter,
    _extract_gsm8k_final_answer,
    _grade_gsm8k_items,
    _load_gsm8k_expected_answers,
    _normalize_number,
)
from llmtrace.benchmarks.models import (
    BenchmarkItemResult,
    BenchmarkRunResult,
    ItemStatus,
    TaskAttempt,
    TaskStatus,
    aggregate_item_results,
    compute_item_aggregate_score,
    item_aggregate_summary,
)
from llmtrace.reporting.benchmark_mapper import build_benchmark_report_section

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter():
    return LmEvalAdapter()


def _make_item(item_id: str, score: float, status: ItemStatus = ItemStatus.GRADED, **kwargs):
    """Build a BenchmarkItemResult with minimal boilerplate."""
    kw: dict = {
        "item_id": item_id,
        "task_id": "t",
        "attempt_id": "a",
        "status": status,
        "raw_score": score,
        "normalized_score": score,
    }
    # error_message required for FAILURE / UNGRADABLE
    if status in (ItemStatus.FAILURE, ItemStatus.UNGRADABLE) and "error_message" not in kwargs:
        kw["error_message"] = f"item {item_id} {status.value}"
    kw.update(kwargs)
    return BenchmarkItemResult(**kw)


# ===================================================================
# 1. ItemAggregateResult — fixed denominator
# ===================================================================


class TestItemAggregateFixedDenominator:
    """Score = sum / planned_items, never shrinks on failure/ungradable."""

    def test_8_of_8_correct(self):
        items = [_make_item(f"i{i}", 1.0) for i in range(8)]
        agg = aggregate_item_results(items, planned_item_count=8)
        assert agg.planned_item_count == 8
        assert agg.graded_item_count == 8
        assert agg.correct_count == 8
        assert agg.wrong_count == 0
        assert agg.failure_count == 0
        assert agg.ungradable_count == 0
        assert agg.coverage == 1.0
        assert agg.execution_coverage == 1.0
        assert agg.raw_score == 1.0
        assert agg.normalized_score == 1.0

    def test_7_of_8_correct(self):
        """7 correct + 1 wrong = 0.875"""
        items = [_make_item(f"i{i}", 1.0) for i in range(7)] + [_make_item("i8", 0.0)]
        agg = aggregate_item_results(items, planned_item_count=8)
        assert agg.planned_item_count == 8
        assert agg.graded_item_count == 8
        assert agg.correct_count == 7
        assert agg.wrong_count == 1
        assert agg.normalized_score == 7.0 / 8.0

    def test_6_correct_1_wrong_1_failure(self):
        """6 correct + 1 wrong + 1 failure → score = 6/8 = 0.75"""
        items = (
            [_make_item(f"i{i}", 1.0) for i in range(6)]
            + [_make_item("i7", 0.0, ItemStatus.GRADED)]
            + [_make_item("i8", 0.0, ItemStatus.FAILURE, error_message="timeout")]
        )
        agg = aggregate_item_results(items, planned_item_count=8)
        assert agg.planned_item_count == 8
        assert agg.graded_item_count == 7
        assert agg.failure_count == 1
        assert agg.ungradable_count == 0
        assert agg.correct_count == 6
        assert agg.wrong_count == 1
        assert agg.coverage == 7.0 / 8.0
        assert agg.execution_coverage == 8.0 / 8.0
        # Must be 6/8, NOT 6/7
        assert agg.raw_score == 6.0 / 8.0
        assert agg.normalized_score == 0.75

    def test_6_correct_1_wrong_1_ungradable(self):
        """6 correct + 1 wrong + 1 ungradable → score = 6/8 = 0.75"""
        items = (
            [_make_item(f"i{i}", 1.0) for i in range(6)]
            + [_make_item("i7", 0.0, ItemStatus.GRADED)]
            + [_make_item("i8", 0.0, ItemStatus.UNGRADABLE, error_message="no parse")]
        )
        agg = aggregate_item_results(items, planned_item_count=8)
        assert agg.ungradable_count == 1
        assert agg.normalized_score == 0.75

    def test_all_correct_with_planned_count(self):
        items = [_make_item(f"i{i}", 1.0) for i in range(8)]
        agg = aggregate_item_results(items, planned_item_count=8)
        assert agg.normalized_score == 1.0

    def test_empty_items_zero_planned(self):
        agg = aggregate_item_results([], planned_item_count=0)
        assert agg.normalized_score == 0.0
        assert agg.coverage == 0.0

    def test_coverage_excludes_failures_from_graded(self):
        """Failures are NOT in graded items, so coverage < 1."""
        items = [
            _make_item("i1", 1.0),
            _make_item("i2", 0.0, ItemStatus.FAILURE, error_message="err"),
        ]
        agg = aggregate_item_results(items, planned_item_count=2)
        assert agg.coverage == 0.5
        assert agg.execution_coverage == 1.0  # failure counts as "executed"


# ===================================================================
# 2. BenchmarkItemResult status validators
# ===================================================================


class TestItemResultValidators:
    """Block illegal status ↔ score combinations."""

    def test_graded_with_error_message_rejected(self):
        with pytest.raises(ValueError, match="GRADED.*error_message"):
            BenchmarkItemResult(
                item_id="i1",
                task_id="t",
                attempt_id="a",
                status=ItemStatus.GRADED,
                raw_score=1.0,
                error_message="should not be here",
            )

    def test_failure_with_positive_score_rejected(self):
        with pytest.raises(ValueError, match="FAILURE.*raw_score=0.0"):
            BenchmarkItemResult(
                item_id="i1",
                task_id="t",
                attempt_id="a",
                status=ItemStatus.FAILURE,
                raw_score=0.7,
                error_message="fail",
            )

    def test_ungradable_with_positive_score_rejected(self):
        with pytest.raises(ValueError, match="UNGRADABLE.*raw_score=0.0"):
            BenchmarkItemResult(
                item_id="i1",
                task_id="t",
                attempt_id="a",
                status=ItemStatus.UNGRADABLE,
                raw_score=1.0,
                error_message="no parse",
            )

    def test_failure_without_error_message_rejected(self):
        with pytest.raises(ValueError, match="FAILURE.*error_message"):
            BenchmarkItemResult(
                item_id="i1",
                task_id="t",
                attempt_id="a",
                status=ItemStatus.FAILURE,
            )

    def test_ungradable_without_error_message_rejected(self):
        with pytest.raises(ValueError, match="UNGRADABLE.*error_message"):
            BenchmarkItemResult(
                item_id="i1",
                task_id="t",
                attempt_id="a",
                status=ItemStatus.UNGRADABLE,
            )

    def test_graded_valid_state(self):
        item = BenchmarkItemResult(
            item_id="i1",
            task_id="t",
            attempt_id="a",
            status=ItemStatus.GRADED,
            raw_score=0.5,
            normalized_score=0.5,
        )
        assert item.raw_score == 0.5

    def test_failure_valid_state(self):
        item = BenchmarkItemResult(
            item_id="i1",
            task_id="t",
            attempt_id="a",
            status=ItemStatus.FAILURE,
            raw_score=0.0,
            error_message="provider timeout",
        )
        assert item.status == ItemStatus.FAILURE
        assert item.error_message == "provider timeout"

    def test_score_boundaries(self):
        BenchmarkItemResult(item_id="i1", task_id="t", attempt_id="a", raw_score=0.0)
        BenchmarkItemResult(item_id="i1", task_id="t", attempt_id="a", raw_score=1.0)
        with pytest.raises(ValueError):
            BenchmarkItemResult(item_id="i1", task_id="t", attempt_id="a", raw_score=-0.1)
        with pytest.raises(ValueError):
            BenchmarkItemResult(item_id="i1", task_id="t", attempt_id="a", raw_score=1.1)


# ===================================================================
# 3. TaskAttempt item validators
# ===================================================================


class TestTaskAttemptItemValidators:
    """Duplicate item_id, cross-attempt, evidence closure."""

    def test_duplicate_item_id_rejected(self):
        item1 = _make_item("dup", 1.0, attempt_id="att-1", task_id="t")
        item2 = _make_item("dup", 0.0, attempt_id="att-1", task_id="t")
        with pytest.raises(ValueError, match="Duplicate item_id"):
            TaskAttempt(
                attempt_id="att-1",
                task_id="t",
                item_results=[item1, item2],
                source_id="s",
                source_revision="r",
                suite_id="s",
                suite_version="v",
                adapter_id="a",
                adapter_version="v",
            )

    def test_unique_item_ids_accepted(self):
        items = [_make_item(f"item-{i:03d}", 1.0, attempt_id="att-1", task_id="t") for i in range(8)]
        attempt = TaskAttempt(
            attempt_id="att-1",
            task_id="t",
            item_results=items,
            source_id="s",
            source_revision="r",
            suite_id="s",
            suite_version="v",
            adapter_id="a",
            adapter_version="v",
        )
        assert len(attempt.item_results) == 8

    def test_cross_attempt_item_rejected(self):
        """Item with attempt_id != parent attempt_id must be rejected."""
        item = _make_item("i1", 1.0, attempt_id="OTHER-ATTEMPT", task_id="t")
        with pytest.raises(ValueError, match="attempt_id"):
            TaskAttempt(
                attempt_id="att-1",
                task_id="t",
                item_results=[item],
                source_id="s",
                source_revision="r",
                suite_id="s",
                suite_version="v",
                adapter_id="a",
                adapter_version="v",
            )

    def test_task_id_mismatch_rejected(self):
        """Item with task_id != parent task_id must be rejected."""
        item = _make_item("i1", 1.0, attempt_id="att-1", task_id="WRONG-TASK")
        with pytest.raises(ValueError, match="task_id"):
            TaskAttempt(
                attempt_id="att-1",
                task_id="t",
                item_results=[item],
                source_id="s",
                source_revision="r",
                suite_id="s",
                suite_version="v",
                adapter_id="a",
                adapter_version="v",
            )

    def test_item_evidence_not_in_attempt_rejected(self):
        """Item referencing evidence not in parent attempt must be rejected."""
        item = BenchmarkItemResult(
            item_id="i1",
            task_id="t",
            attempt_id="att-1",
            evidence_refs=["00000000-0000-0000-0000-000000000999"],
        )
        with pytest.raises(ValueError, match="evidence.*not in"):
            TaskAttempt(
                attempt_id="att-1",
                task_id="t",
                item_results=[item],
                evidence_refs=["00000000-0000-0000-0000-000000000001"],
                source_id="s",
                source_revision="r",
                suite_id="s",
                suite_version="v",
                adapter_id="a",
                adapter_version="v",
            )


# ===================================================================
# 4. GSM8K answer extraction
# ===================================================================


class TestGsm8kAnswerExtraction:
    def test_extract_simple(self):
        assert _extract_gsm8k_final_answer("#### 42") == "42"

    def test_extract_negative(self):
        assert _extract_gsm8k_final_answer("#### -5") == "-5"

    def test_extract_with_comma(self):
        assert _extract_gsm8k_final_answer("#### 1,234") == "1,234"

    def test_extract_no_answer(self):
        assert _extract_gsm8k_final_answer("No answer") is None

    def test_extract_empty(self):
        assert _extract_gsm8k_final_answer("") is None


class TestNumberNormalization:
    def test_integer(self):
        assert _normalize_number("42") == "42"

    def test_decimal(self):
        assert _normalize_number("3.14") == "3.14"

    def test_remove_comma(self):
        assert _normalize_number("1,234") == "1234"

    def test_trailing_zero(self):
        assert _normalize_number("5.0") == "5"


# ===================================================================
# 5. GSM8K item grading
# ===================================================================


class TestGsm8kItemGrading:
    def _sample(self, evidence_id: str, response_text: str, status: str = "success"):
        return {"evidence_id": evidence_id, "response_text": response_text, "prompt": "", "status": status}

    def test_loaded_answers_count(self):
        answers = _load_gsm8k_expected_answers()
        assert len(answers) == 8

    def test_all_8_items_produced(self):
        responses = [
            self._sample(f"00000000-0000-0000-0000-0000000000{i:02d}", f"Reasoning\n#### {42 + i}") for i in range(8)
        ]
        items = _grade_gsm8k_items(responses, "attempt-1", "gsm8k_subset")
        assert len(items) == 8
        for i, item in enumerate(items):
            assert item.item_id == f"item-{i + 1:03d}"
            assert item.attempt_id == "attempt-1"
            assert item.task_id == "gsm8k_subset"
            assert len(item.evidence_refs) == 1

    def test_wrong_answer_isolated(self):
        """item-001 correct, item-002 wrong, item-003 correct."""
        responses = [
            self._sample("00000000-0000-0000-0000-000000000001", "#### 42"),
            self._sample("00000000-0000-0000-0000-000000000002", "#### 999"),
            self._sample("00000000-0000-0000-0000-000000000003", "#### 44"),
        ]
        items = _grade_gsm8k_items(responses, "a", "t")
        assert all(item.status == ItemStatus.GRADED for item in items)
        # Score assertions depend on expected answers loaded from JSON.
        # item-002 is graded 0 because 999 != expected[1].
        assert items[1].raw_score == 0.0
        assert items[1].normalized_score == 0.0

    def test_ungradable_no_answer_pattern(self):
        responses = [self._sample("00000000-0000-0000-0000-00000000000a", "No #### here")]
        items = _grade_gsm8k_items(responses, "a", "t")
        assert items[0].status == ItemStatus.UNGRADABLE
        assert items[0].error_message is not None
        assert "####" in items[0].error_message

    def test_failure_item_mapped_correctly(self):
        """A bridge-level failure sample must produce ItemStatus.FAILURE."""
        responses = [
            self._sample(
                "00000000-0000-0000-0000-000000000fff",
                "",
                status="failure",
            )
            | {"failure_message": "provider timeout", "failure_error_code": "PROVIDER_EXCEPTION"},
        ]
        items = _grade_gsm8k_items(responses, "a", "t")
        assert items[0].status == ItemStatus.FAILURE
        assert items[0].raw_score == 0.0
        assert items[0].normalized_score == 0.0
        assert "provider timeout" in items[0].error_message

    def test_item_ids_deterministic(self):
        responses = [self._sample(f"00000000-0000-0000-0000-00000000000{i}", "#### 1") for i in range(3)]
        items1 = _grade_gsm8k_items(responses, "a", "t")
        items2 = _grade_gsm8k_items(responses, "a", "t")
        assert [i.item_id for i in items1] == [i.item_id for i in items2]


# ===================================================================
# 6. GSM8K full pipeline (uses FakeProvider)
# ===================================================================


class TestGsm8kFullPipeline:
    async def test_pipeline_produces_8_items(self, adapter):
        from tests.adapters.conftest import FakeProvider

        tasks = adapter.list_tasks()
        gsm_spec = next(t for t in tasks if t.task_id == "gsm8k_subset")
        provider = FakeProvider(
            response_map={
                "Janet's ducks lay 16 eggs per day": "#### 18",
                "A robe takes 2 bolts of blue fiber": "#### 3",
                "Josh decides to try flipping a house": "#### 70000",
                "James decides to run 3 sprints": "#### 540",
                "Every day, Wendi feeds each of her chickens": "#### 20",
                "Kylar went to the store to buy glasses": "#### 64",
                "Toulouse has twice as many sheep as Charleston": "#### 260",
                "Carla is downloading a 200 GB file": "#### 160",
            },
        )

        attempt = await adapter.run_task(gsm_spec, provider)
        assert len(attempt.item_results) == 8

    async def test_pipeline_each_item_has_unique_evidence(self, adapter):
        from tests.adapters.conftest import FakeProvider

        tasks = adapter.list_tasks()
        gsm_spec = next(t for t in tasks if t.task_id == "gsm8k_subset")
        provider = FakeProvider(
            response_map={
                "Janet's ducks lay 16 eggs per day": "#### 18",
                "A robe takes 2 bolts of blue fiber": "#### 3",
                "Josh decides to try flipping a house": "#### 70000",
                "James decides to run 3 sprints": "#### 540",
                "Every day, Wendi feeds each of her chickens": "#### 20",
                "Kylar went to the store to buy glasses": "#### 64",
                "Toulouse has twice as many sheep as Charleston": "#### 260",
                "Carla is downloading a 200 GB file": "#### 160",
            },
        )

        attempt = await adapter.run_task(gsm_spec, provider)
        evidence_ids = [ir.evidence_refs[0] for ir in attempt.item_results if ir.evidence_refs]
        assert len(evidence_ids) == 8
        assert len(set(evidence_ids)) == 8


# ===================================================================
# 7. Provider failure isolation (FakeProvider)
# ===================================================================


class TestProviderFailureIsolation:
    """A single provider failure must NOT abort remaining items."""

    async def test_failure_isolation_from_fake_provider(self, adapter):
        """Item 3 fails (HTTP 500), all other items continue."""

        tasks = adapter.list_tasks()
        gsm_spec = next(t for t in tasks if t.task_id == "gsm8k_subset")

        # We need FakeProvider to fail on a specific call, using fail_with_http_status
        # on a specific call number.
        # Since FakeProvider doesn't directly support per-call failure mode,
        # we use fail_with_http_status=500 to make ALL responses fail via evidence check.
        # For true per-item isolation, we need a custom provider.

        # Instead, test via manual composition: create a FakeProvider that returns
        # HTTP 500 evidence on the 3rd call.
        class SelectiveFailingProvider:
            """Provider that fails on a specific call index."""

            def __init__(self):
                self.call_count = 0
                self.evidence = []

            async def complete(self, model, messages, *, options=None):

                self.call_count += 1
                fail = self.call_count == 3  # fail on 3rd item
                from llmtrace.models.evidence import HTTPEvidence

                evidence_id = uuid.uuid4()
                ev = HTTPEvidence(
                    evidence_id=evidence_id,
                    evidence_type="smoke_test",
                    request_method="POST",
                    request_url_redacted="https://fake/",
                    request_path="/",
                    request_headers_redacted={},
                    request_body_redacted={},
                    request_model=model,
                    response_model=model,
                    response_text="" if fail else "#### 42",
                    http_status=500 if fail else 200,
                    total_latency_ms=100.0,
                )
                self.evidence.append(ev)
                return ev

        provider = SelectiveFailingProvider()
        attempt = await adapter.run_task(gsm_spec, provider)

        # All 8 items should exist
        assert len(attempt.item_results) == 8

        # Item 3 (idx 2) should be FAILURE
        failed_items = [it for it in attempt.item_results if it.status == ItemStatus.FAILURE]
        assert len(failed_items) == 1
        assert failed_items[0].item_id == "item-003"

        # Other items should be GRADED or UNGRADABLE (not FAILURE)
        non_failed = [it for it in attempt.item_results if it.status != ItemStatus.FAILURE]
        assert len(non_failed) == 7

        # Failed item must have evidence
        assert len(failed_items[0].evidence_refs) == 1
        assert len(failed_items[0].error_message) > 0


# ===================================================================
# 8. normalize_result — item-derived GradeResult with cross-check
# ===================================================================


class TestNormalizeResultItemDerived:
    """GradeResult must derive from ItemAggregateResult; lm-eval is cross-check only."""

    def test_normalize_derives_from_items(self, adapter):
        """7 correct + 1 wrong → score = 0.875 from items, cross-checked with lm-eval."""
        items = [_make_item(f"item-{i:03d}", 1.0) for i in range(7)] + [_make_item("item-008", 0.0)]
        raw_result = {
            "results": {"exact_match": 0.875},
            "evidence_ids": [],
            "task_name": "gsm8k_subset",
            "attempt_id": "att-1",
            "item_results": [it.model_dump() for it in items],
            "planned_item_count": 8,
        }
        grade = adapter.normalize_result(raw_result)
        assert grade.normalized_score == 0.875
        # Verify metadata contains aggregate info
        assert grade.metadata.get("planned_item_count") == 8
        assert grade.metadata.get("correct_count") == 7
        assert grade.metadata.get("wrong_count") == 1
        assert grade.metadata.get("lm_eval_cross_check_pass") is True

    def test_normalize_mismatch_raises(self, adapter):
        """If item aggregate != lm-eval metric, raise ValueError."""
        items = [_make_item(f"item-{i:03d}", 1.0) for i in range(8)]
        raw_result = {
            "results": {"exact_match": 0.5},  # deliberately wrong
            "evidence_ids": [],
            "task_name": "gsm8k_subset",
            "attempt_id": "att-1",
            "item_results": [it.model_dump() for it in items],
            "planned_item_count": 8,
        }
        with pytest.raises(ValueError, match="LM_EVAL_ITEM_AGGREGATE_MISMATCH"):
            adapter.normalize_result(raw_result)

    def test_normalize_falls_back_to_lm_eval_without_items(self, adapter):
        """When no item_results, fall back to lm-eval metric."""
        raw_result = {
            "results": {"exact_match": 0.5},
            "evidence_ids": [],
            "task_name": "gsm8k_subset",
            "attempt_id": "att-1",
        }
        grade = adapter.normalize_result(raw_result)
        assert grade.normalized_score == 0.5

    def test_normalize_6_correct_1_wrong_1_failure(self, adapter):
        """6 correct + 1 wrong + 1 failure → score = 0.75"""
        items = (
            [_make_item(f"item-{i:03d}", 1.0) for i in range(6)]
            + [_make_item("item-007", 0.0)]
            + [_make_item("item-008", 0.0, ItemStatus.FAILURE, error_message="timeout")]
        )
        raw_result = {
            "results": {"exact_match": 0.75},
            "evidence_ids": [],
            "task_name": "gsm8k_subset",
            "attempt_id": "att-1",
            "item_results": [it.model_dump() for it in items],
            "planned_item_count": 8,
        }
        grade = adapter.normalize_result(raw_result)
        assert grade.normalized_score == 0.75
        assert grade.metadata.get("failure_count") == 1
        assert grade.metadata.get("graded_item_count") == 7
        assert grade.metadata.get("coverage") == 7.0 / 8.0


# ===================================================================
# 9. Reporting — item-level display in JSON
# ===================================================================


class TestItemLevelReporting:
    """Item-level data appears in JSON reports."""

    async def test_report_section_contains_items(self, adapter):
        from tests.adapters.conftest import FakeProvider

        tasks = adapter.list_tasks()
        gsm_spec = next(t for t in tasks if t.task_id == "gsm8k_subset")
        provider = FakeProvider(
            response_map={
                "Janet's ducks lay 16 eggs per day": "#### 18",
                "A robe takes 2 bolts of blue fiber": "#### 3",
                "Josh decides to try flipping a house": "#### 70000",
                "James decides to run 3 sprints": "#### 540",
                "Every day, Wendi feeds each of her chickens": "#### 20",
                "Kylar went to the store to buy glasses": "#### 64",
                "Toulouse has twice as many sheep as Charleston": "#### 260",
                "Carla is downloading a 200 GB file": "#### 160",
            },
        )

        attempt = await adapter.run_task(gsm_spec, provider)
        raw_result = {
            "results": {"exact_match": 1.0},
            "evidence_ids": list(attempt.evidence_refs),
            "task_name": "gsm8k_subset",
            "attempt_id": attempt.attempt_id,
            "item_results": [ir.model_dump() for ir in attempt.item_results],
            "planned_item_count": 8,
        }
        grade = adapter.normalize_result(raw_result)

        run_result = BenchmarkRunResult(
            run_id=str(uuid.uuid4()),
            task_attempts=[attempt],
            grade_results=[grade],
            evidence_refs=attempt.evidence_refs,
            source_id=attempt.source_id,
            source_revision=attempt.source_revision,
            suite_id=attempt.suite_id,
            suite_version=attempt.suite_version,
            adapter_id=adapter.adapter_id,
            adapter_version=adapter.adapter_version,
        )

        plan = adapter.build_plan(
            attempt.suite_id,
            attempt.suite_version,
            attempt.source_id,
            attempt.source_revision,
            ["gsm8k_subset"],
        )
        section = build_benchmark_report_section(plan, run_result)

        assert len(section.tasks) == 1
        task = section.tasks[0]
        assert len(task.items) == 8
        first_item = task.items[0]
        assert first_item.item_id == "item-001"
        assert first_item.status in ("graded", "ungradable", "failure")
        assert len(first_item.evidence_refs) == 1

    def test_report_section_no_items_for_smoke(self, adapter):
        """Smoke task should have empty items list."""
        attempt = TaskAttempt(
            attempt_id="smoke-1",
            task_id="llmtrace_smoke",
            source_id="lm-eval",
            source_revision="0000000-smoke",
            suite_id="llmtrace_smoke",
            suite_version="1.0.0",
            status=TaskStatus.SUCCESS,
            adapter_id="lm-eval",
            adapter_version="0.4.12",
            evidence_refs=[],
            metadata={"llmtrace_smoke_task": True},
        )

        run_result = BenchmarkRunResult(
            run_id=str(uuid.uuid4()),
            task_attempts=[attempt],
            grade_results=[],
            evidence_refs=[],
            source_id="lm-eval",
            source_revision="0000000-smoke",
            suite_id="llmtrace_smoke",
            suite_version="1.0.0",
            adapter_id="lm-eval",
            adapter_version="0.4.12",
        )

        plan = adapter.build_plan(
            run_result.suite_id,
            run_result.suite_version,
            run_result.source_id,
            run_result.source_revision,
            ["llmtrace_smoke"],
        )
        section = build_benchmark_report_section(plan, run_result)
        assert len(section.tasks) == 1
        assert section.tasks[0].items == []

    def test_report_json_round_trip(self):
        from llmtrace.reporting.benchmark_models import ItemReportItem

        item = ItemReportItem(
            item_id="item-001",
            status="graded",
            raw_score=1.0,
            normalized_score=1.0,
            grader_id="exact_match",
            evidence_refs=["00000000-0000-0000-0000-000000000001"],
            metadata={"correct": True},
        )
        d = item.model_dump(mode="json")
        restored = ItemReportItem(**d)
        assert restored.item_id == "item-001"
        assert restored.raw_score == 1.0
        assert restored.metadata == {"correct": True}


# ===================================================================
# 10. HTML XSS escaping — real renderer
# ===================================================================


class TestHtmlXssEscaping:
    """HTML must escape malicious metadata through the real Jinja renderer."""

    def test_html_escapes_xss_in_item_metadata(self, adapter, tmp_path):
        """Malicious metadata must be HTML-escaped in rendered output.

        The HTML template currently only displays item error_message and status.
        Metadata is not rendered inline, so this test verifies that items with
        XSS-laden metadata do NOT inject raw <script> tags via metadata routes.
        """
        from datetime import UTC, datetime

        from llmtrace.config import AuditConfig, AuthStyle, Protocol
        from llmtrace.models.audit import AuditResult, RiskLevel
        from llmtrace.reporting.benchmark_mapper import build_benchmark_report_section
        from llmtrace.reporting.html_report import generate_html_report

        suite_id = "llmtrace-core-benchmarks"
        suite_version = "0.1.0"
        source_id = "lm-eval-harness"
        source_revision = "9d05167"

        items = [
            BenchmarkItemResult(
                item_id="item-001",
                task_id="llmtrace_smoke",
                attempt_id="att-1",
                status=ItemStatus.GRADED,
                raw_score=1.0,
                normalized_score=1.0,
                metadata={"extracted_answer": "<script>alert(1)</script>"},
            ),
        ]
        attempt = TaskAttempt(
            attempt_id="att-1",
            task_id="llmtrace_smoke",
            source_id=source_id,
            source_revision=source_revision,
            suite_id=suite_id,
            suite_version=suite_version,
            adapter_id=adapter.adapter_id,
            adapter_version=adapter.adapter_version,
            status=TaskStatus.SUCCESS,
            evidence_refs=[],
            item_results=items,
        )
        run_result = BenchmarkRunResult(
            run_id=str(uuid.uuid4()),
            task_attempts=[attempt],
            grade_results=[],
            evidence_refs=[],
            source_id=source_id,
            source_revision=source_revision,
            suite_id=suite_id,
            suite_version=suite_version,
            adapter_id=adapter.adapter_id,
            adapter_version=adapter.adapter_version,
        )
        plan = adapter.build_plan(suite_id, suite_version, source_id, source_revision, ["llmtrace_smoke"])
        section = build_benchmark_report_section(plan, run_result)

        audit = AuditResult(
            config=AuditConfig(
                protocol=Protocol.OPENAI,
                base_url="https://api.example.com",
                model="test",
                api_key_env="K",
                auth_style=AuthStyle.BEARER,
                repeat_count=1,
                timeout=30.0,
                max_output_tokens=100,
                check_streaming=False,
            ),
            evidence=[],
            findings=[],
            risk_level=RiskLevel.INCONCLUSIVE,
            schema_fingerprints=[],
            model_list=[],
            start_time=datetime(2026, 8, 1, tzinfo=UTC),
            end_time=datetime(2026, 8, 1, 1, tzinfo=UTC),
            llmtrace_version="0.3.0",
            python_version="3.12",
            platform="darwin",
            report_id="xss-test",
            content_hash="",
        )

        out_path = tmp_path / "report.html"
        generate_html_report(audit, out_path, benchmark_sections=[section])

        html = out_path.read_text()
        # Raw <script> tag must not appear anywhere in the HTML
        assert "<script>alert(1)</script>" not in html

    def test_html_escapes_xss_in_error_message(self, adapter, tmp_path):
        """Malicious error_message must be HTML-escaped."""
        from datetime import UTC, datetime

        from llmtrace.config import AuditConfig, AuthStyle, Protocol
        from llmtrace.models.audit import AuditResult, RiskLevel
        from llmtrace.reporting.benchmark_mapper import build_benchmark_report_section
        from llmtrace.reporting.html_report import generate_html_report

        suite_id = "llmtrace-core-benchmarks"
        suite_version = "0.1.0"
        source_id = "lm-eval-harness"
        source_revision = "9d05167"

        items = [
            BenchmarkItemResult(
                item_id="item-001",
                task_id="llmtrace_smoke",
                attempt_id="att-1",
                status=ItemStatus.FAILURE,
                raw_score=0.0,
                normalized_score=0.0,
                error_message="<script>alert('xss')</script>",
            ),
        ]
        attempt = TaskAttempt(
            attempt_id="att-1",
            task_id="llmtrace_smoke",
            source_id=source_id,
            source_revision=source_revision,
            suite_id=suite_id,
            suite_version=suite_version,
            adapter_id=adapter.adapter_id,
            adapter_version=adapter.adapter_version,
            status=TaskStatus.SUCCESS,
            evidence_refs=[],
            item_results=items,
        )
        run_result = BenchmarkRunResult(
            run_id=str(uuid.uuid4()),
            task_attempts=[attempt],
            grade_results=[],
            evidence_refs=[],
            source_id=source_id,
            source_revision=source_revision,
            suite_id=suite_id,
            suite_version=suite_version,
            adapter_id=adapter.adapter_id,
            adapter_version=adapter.adapter_version,
        )
        plan = adapter.build_plan(suite_id, suite_version, source_id, source_revision, ["llmtrace_smoke"])
        section = build_benchmark_report_section(plan, run_result)

        audit = AuditResult(
            config=AuditConfig(
                protocol=Protocol.OPENAI,
                base_url="https://api.example.com",
                model="test",
                api_key_env="K",
                auth_style=AuthStyle.BEARER,
                repeat_count=1,
                timeout=30.0,
                max_output_tokens=100,
                check_streaming=False,
            ),
            evidence=[],
            findings=[],
            risk_level=RiskLevel.INCONCLUSIVE,
            schema_fingerprints=[],
            model_list=[],
            start_time=datetime(2026, 8, 1, tzinfo=UTC),
            end_time=datetime(2026, 8, 1, 1, tzinfo=UTC),
            llmtrace_version="0.3.0",
            python_version="3.12",
            platform="darwin",
            report_id="xss-test-2",
            content_hash="",
        )

        out_path = tmp_path / "report2.html"
        generate_html_report(audit, out_path, benchmark_sections=[section])

        html = out_path.read_text()
        assert "<script>alert('xss')</script>" not in html
        assert "&lt;script&gt;" in html


# ===================================================================
# 11. Deprecated helpers — backward compat
# ===================================================================


class TestDeprecatedHelpers:
    """Verify old compute_item_aggregate_score still works for callers."""

    def test_old_compute_still_works(self):
        items = [_make_item("i1", 1.0), _make_item("i2", 0.0)]
        assert compute_item_aggregate_score(items) == 0.5

    def test_item_aggregate_summary(self):
        items = [_make_item("i1", 1.0), _make_item("i2", 0.0, ItemStatus.FAILURE, error_message="err")]
        summary = item_aggregate_summary(items)
        assert summary["total_items"] == 2
        assert summary["failure_count"] == 1
        # With fixed denominator: 1 graded score of 1.0 / 2 planned = 0.5
        assert summary["item_aggregate_score"] == 0.5


# ===================================================================
# 12. GSM8K calibrated score remains None
# ===================================================================


class TestCalibratedScoreRemainsNone:
    """v0.3-A must not introduce calibrated scoring."""

    async def test_calibrated_score_is_none(self, adapter):
        from llmtrace.scoring import (
            CapabilityDimension,
            TaskScoringRegistry,
            TaskScoringSpec,
            aggregate_dimension_score,
        )
        from llmtrace.scoring.policy import CapabilityScoringPolicy
        from tests.adapters.conftest import FakeProvider

        tasks = adapter.list_tasks()
        gsm_spec = next(t for t in tasks if t.task_id == "gsm8k_subset")
        provider = FakeProvider(
            response_map={
                "Janet's ducks lay 16 eggs per day": "#### 18",
                "A robe takes 2 bolts of blue fiber": "#### 3",
                "Josh decides to try flipping a house": "#### 70000",
                "James decides to run 3 sprints": "#### 540",
                "Every day, Wendi feeds each of her chickens": "#### 20",
                "Kylar went to the store to buy glasses": "#### 64",
                "Toulouse has twice as many sheep as Charleston": "#### 260",
                "Carla is downloading a 200 GB file": "#### 160",
            },
        )

        attempt = await adapter.run_task(gsm_spec, provider)
        raw_result = {
            "results": {"exact_match": 1.0},
            "evidence_ids": list(attempt.evidence_refs),
            "task_name": "gsm8k_subset",
            "attempt_id": attempt.attempt_id,
            "item_results": [ir.model_dump() for ir in attempt.item_results],
            "planned_item_count": 8,
        }
        grade = adapter.normalize_result(raw_result)

        run_result = BenchmarkRunResult(
            run_id=str(uuid.uuid4()),
            task_attempts=[attempt],
            grade_results=[grade],
            evidence_refs=attempt.evidence_refs,
            source_id=attempt.source_id,
            source_revision=attempt.source_revision,
            suite_id=attempt.suite_id,
            suite_version=attempt.suite_version,
            adapter_id=adapter.adapter_id,
            adapter_version=adapter.adapter_version,
        )

        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(
                    task_id="gsm8k_subset",
                    dimension=CapabilityDimension.MATH_SCIENCE,
                    task_weight=1.0,
                )
            ]
        )
        policy = CapabilityScoringPolicy.create_v1()
        dim_score = aggregate_dimension_score(
            CapabilityDimension.MATH_SCIENCE,
            [run_result],
            registry,
            policy,
        )
        assert dim_score.calibrated_score is None
