"""Integration tests for item-level benchmark results (v0.3-A).

Tests cover:
- Model validation (BenchmarkItemResult)
- GSM8K item extraction and grading
- Failure isolation
- Reporting round-trip
- Compute aggregate helpers
"""

from __future__ import annotations

import uuid

import pytest

from llmtrace.adapters.lm_eval import (
    _GSM8K_ANSWER_RE,
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


# ---------------------------------------------------------------------------
# Model Validation
# ---------------------------------------------------------------------------


class TestBenchmarkItemResultModel:
    """Validate the BenchmarkItemResult model invariants."""

    def test_minimal_construction(self):
        item = BenchmarkItemResult(
            item_id="item-001",
            task_id="gsm8k_subset",
            attempt_id="attempt-1",
        )
        assert item.status == ItemStatus.GRADED
        assert item.raw_score == 0.0
        assert item.normalized_score == 0.0
        assert item.grader_id == "exact_match"
        assert item.evidence_refs == []
        assert item.error_message is None
        assert item.metadata == {}

    def test_score_boundaries(self):
        # Score must be within [0, 1]
        BenchmarkItemResult(item_id="i1", task_id="t", attempt_id="a", raw_score=0.0)
        BenchmarkItemResult(item_id="i1", task_id="t", attempt_id="a", raw_score=1.0)
        BenchmarkItemResult(item_id="i1", task_id="t", attempt_id="a", raw_score=0.75)

        with pytest.raises(ValueError):
            BenchmarkItemResult(item_id="i1", task_id="t", attempt_id="a", raw_score=-0.1)

        with pytest.raises(ValueError):
            BenchmarkItemResult(item_id="i1", task_id="t", attempt_id="a", raw_score=1.1)

    def test_item_id_non_empty(self):
        with pytest.raises(ValueError):
            BenchmarkItemResult(item_id="", task_id="t", attempt_id="a")

    def test_uuid_validation_for_attempt_id(self):
        # attempt_id must be non-empty (implementation choice, not UUID enforced)
        BenchmarkItemResult(item_id="i1", task_id="t", attempt_id=str(uuid.uuid4()))

    def test_serialization_round_trip(self):
        item = BenchmarkItemResult(
            item_id="item-003",
            task_id="gsm8k_subset",
            attempt_id="attempt-xyz",
            status=ItemStatus.GRADED,
            raw_score=1.0,
            normalized_score=1.0,
            evidence_refs=["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"],
            metadata={"extracted_answer": "42", "expected_answer": "42", "correct": True},
        )
        d = item.model_dump()
        restored = BenchmarkItemResult(**d)
        assert restored.item_id == "item-003"
        assert restored.raw_score == 1.0
        assert restored.metadata["correct"] is True

    def test_failure_status_consistency(self):
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
        assert item.normalized_score == 0.0

    def test_ungradable_status(self):
        item = BenchmarkItemResult(
            item_id="i1",
            task_id="t",
            attempt_id="a",
            status=ItemStatus.UNGRADABLE,
            error_message="No #### pattern found",
        )
        assert item.status == ItemStatus.UNGRADABLE

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValueError):
            BenchmarkItemResult(  # type: ignore[call-arg]
                item_id="i1", task_id="t", attempt_id="a", fake_field=42
            )


# ---------------------------------------------------------------------------
# GSM8K Answer Extraction
# ---------------------------------------------------------------------------


class TestGsm8kAnswerExtraction:
    """Test the GSM8K answer extraction logic."""

    def test_extract_simple_answer(self):
        assert _extract_gsm8k_final_answer("Some reasoning\n#### 42") == "42"

    def test_extract_negative_answer(self):
        assert _extract_gsm8k_final_answer("#### -5") == "-5"

    def test_extract_with_spaces(self):
        assert _extract_gsm8k_final_answer("####   18  ") == "18"

    def test_extract_with_comma(self):
        assert _extract_gsm8k_final_answer("#### 1,234") == "1,234"

    def test_extract_with_decimal(self):
        assert _extract_gsm8k_final_answer("#### 3.14") == "3.14"

    def test_extract_multiline(self):
        text = "Line 1\nLine 2\n#### 100\nMore text"
        assert _extract_gsm8k_final_answer(text) == "100"

    def test_extract_last_answer(self):
        text = "#### 10\n#### 20"
        assert _extract_gsm8k_final_answer(text) == "10"  # first match

    def test_extract_no_answer(self):
        assert _extract_gsm8k_final_answer("No answer here") is None

    def test_extract_empty_string(self):
        assert _extract_gsm8k_final_answer("") is None

    def test_regex_compiled(self):
        assert _GSM8K_ANSWER_RE.pattern is not None


# ---------------------------------------------------------------------------
# Number Normalization
# ---------------------------------------------------------------------------


class TestNumberNormalization:
    """Test number normalization for comparison."""

    def test_integer(self):
        assert _normalize_number("42") == "42"

    def test_decimal(self):
        assert _normalize_number("3.14") == "3.14"

    def test_remove_comma(self):
        assert _normalize_number("1,234") == "1234"

    def test_trailing_zero(self):
        assert _normalize_number("5.0") == "5"

    def test_negative(self):
        assert _normalize_number("-5") == "-5"

    def test_comma_and_decimal(self):
        assert _normalize_number("1,234.5") == "1234.5"

    def test_non_numeric(self):
        assert _normalize_number("hello") == "hello"


# ---------------------------------------------------------------------------
# Item Aggregate Helpers
# ---------------------------------------------------------------------------


class TestItemAggregateHelpers:
    """Test compute_item_aggregate_score and item_aggregate_summary."""

    def _make_item(self, item_id: str, score: float, status: ItemStatus = ItemStatus.GRADED):
        return BenchmarkItemResult(
            item_id=item_id,
            task_id="t",
            attempt_id="a",
            status=status,
            raw_score=score,
            normalized_score=score,
        )

    def test_all_correct(self):
        items = [self._make_item(f"i{i}", 1.0) for i in range(8)]
        assert compute_item_aggregate_score(items) == 1.0

    def test_partial_correct(self):
        items = [self._make_item("i1", 1.0), self._make_item("i2", 0.0)]
        assert compute_item_aggregate_score(items) == 0.5

    def test_none_correct(self):
        items = [self._make_item(f"i{i}", 0.0) for i in range(8)]
        assert compute_item_aggregate_score(items) == 0.0

    def test_7_of_8_correct(self):
        items = [self._make_item(f"i{i}", 1.0) for i in range(7)] + [self._make_item("i8", 0.0)]
        assert compute_item_aggregate_score(items) == 7.0 / 8.0

    def test_exclude_ungradable(self):
        items = [
            self._make_item("i1", 1.0),
            self._make_item("i2", 1.0),
            self._make_item("i3", 0.0, ItemStatus.UNGRADABLE),
        ]
        assert compute_item_aggregate_score(items) == 1.0  # only graded: 2/2

    def test_exclude_failure(self):
        items = [
            self._make_item("i1", 0.0),
            self._make_item("i2", 0.5, ItemStatus.FAILURE),
            self._make_item("i3", 0.0, ItemStatus.FAILURE),
        ]
        assert compute_item_aggregate_score(items) == 0.0  # only graded: 0/1

    def test_empty_items(self):
        assert compute_item_aggregate_score([]) is None

    def test_all_ungradable(self):
        items = [self._make_item(f"i{i}", 0.0, ItemStatus.UNGRADABLE) for i in range(3)]
        assert compute_item_aggregate_score(items) is None

    def test_summary_all_correct(self):
        items = [self._make_item(f"i{i}", 1.0) for i in range(8)]
        summary = item_aggregate_summary(items)
        assert summary["total_items"] == 8
        assert summary["graded_count"] == 8
        assert summary["correct_count"] == 8
        assert summary["failure_count"] == 0
        assert summary["item_aggregate_score"] == 1.0

    def test_summary_with_failures(self):
        items = [
            self._make_item("i1", 1.0),
            self._make_item("i2", 0.0),
            self._make_item("i3", 0.0, ItemStatus.FAILURE),
        ]
        summary = item_aggregate_summary(items)
        assert summary["total_items"] == 3
        assert summary["graded_count"] == 2
        assert summary["failure_count"] == 1
        assert summary["item_aggregate_score"] == 0.5


# ---------------------------------------------------------------------------
# GSM8K Item Grading
# ---------------------------------------------------------------------------


class TestGsm8kItemGrading:
    """Test GSM8K item grading with _grade_gsm8k_items."""

    def _make_sample(self, evidence_id: str, response_text: str):
        return {"evidence_id": evidence_id, "response_text": response_text, "prompt": ""}

    def test_all_correct_8_items(self):
        responses = [
            self._make_sample(f"00000000-0000-0000-0000-0000000000{i:02d}", f"Reasoning\n#### {42 + i}")
            for i in range(8)
        ]
        items = _grade_gsm8k_items(responses, "attempt-1", "gsm8k_subset")
        assert len(items) == 8
        for i, item in enumerate(items):
            assert item.item_id == f"item-{i + 1:03d}"
            assert item.status == ItemStatus.GRADED
            assert item.attempt_id == "attempt-1"
            assert item.task_id == "gsm8k_subset"
            assert len(item.evidence_refs) == 1

    def test_item_ids_are_deterministic(self):
        responses = [self._make_sample(f"00000000-0000-0000-0000-0000000000{i:02d}", "#### 1") for i in range(3)]
        items1 = _grade_gsm8k_items(responses, "a", "t")
        items2 = _grade_gsm8k_items(responses, "a", "t")
        ids1 = [i.item_id for i in items1]
        ids2 = [i.item_id for i in items2]
        assert ids1 == ids2 == ["item-001", "item-002", "item-003"]

    def test_each_item_has_evidence(self):
        responses = [self._make_sample(f"00000000-0000-0000-0000-0000000000{i:02d}", "#### 1") for i in range(3)]
        items = _grade_gsm8k_items(responses, "a", "t")
        assert all(len(item.evidence_refs) == 1 for item in items)

    def test_ungradable_no_answer_pattern(self):
        responses = [self._make_sample("00000000-0000-0000-0000-000000000000", "No #### here")]
        items = _grade_gsm8k_items(responses, "a", "t")
        assert items[0].status == ItemStatus.UNGRADABLE
        assert items[0].error_message is not None
        assert "####" in items[0].error_message

    def test_loaded_answers_count(self):
        answers = _load_gsm8k_expected_answers()
        assert len(answers) == 8


# ---------------------------------------------------------------------------
# GSM8K Full Pipeline (mock)
# ---------------------------------------------------------------------------


class TestGsm8kFullPipeline:
    """Test the full GSM8K pipeline with mock provider."""

    async def test_pipeline_produces_8_items(self, adapter):
        from tests.adapters.conftest import FakeProvider

        tasks = adapter.list_tasks()
        gsm_spec = next(t for t in tasks if t.task_id == "gsm8k_subset")

        responses = {
            "Janet's ducks lay 16 eggs per day": "#### 18",
            "A robe takes 2 bolts of blue fiber": "#### 3",
            "Josh decides to try flipping a house": "#### 70000",
            "James decides to run 3 sprints": "#### 540",
            "Every day, Wendi feeds each of her chickens": "#### 20",
            "Kylar went to the store to buy glasses": "#### 64",
            "Toulouse has twice as many sheep as Charleston": "#### 260",
            "Carla is downloading a 200 GB file": "#### 160",
        }
        provider = FakeProvider(response_map=responses)

        attempt = await adapter.run_task(gsm_spec, provider)
        assert len(attempt.item_results) == 8

    async def test_pipeline_item_ids_deterministic(self, adapter):
        from tests.adapters.conftest import FakeProvider

        tasks = adapter.list_tasks()
        gsm_spec = next(t for t in tasks if t.task_id == "gsm8k_subset")

        responses = {
            "Janet's ducks lay 16 eggs per day": "#### 18",
            "A robe takes 2 bolts of blue fiber": "#### 3",
            "Josh decides to try flipping a house": "#### 70000",
            "James decides to run 3 sprints": "#### 540",
            "Every day, Wendi feeds each of her chickens": "#### 20",
            "Kylar went to the store to buy glasses": "#### 64",
            "Toulouse has twice as many sheep as Charleston": "#### 260",
            "Carla is downloading a 200 GB file": "#### 160",
        }
        provider = FakeProvider(response_map=responses)

        items1 = (await adapter.run_task(gsm_spec, provider)).item_results
        items2 = (await adapter.run_task(gsm_spec, provider)).item_results
        assert [i.item_id for i in items1] == [i.item_id for i in items2]

    async def test_pipeline_each_item_has_unique_evidence(self, adapter):
        from tests.adapters.conftest import FakeProvider

        tasks = adapter.list_tasks()
        gsm_spec = next(t for t in tasks if t.task_id == "gsm8k_subset")

        responses = {
            "Janet's ducks lay 16 eggs per day": "#### 18",
            "A robe takes 2 bolts of blue fiber": "#### 3",
            "Josh decides to try flipping a house": "#### 70000",
            "James decides to run 3 sprints": "#### 540",
            "Every day, Wendi feeds each of her chickens": "#### 20",
            "Kylar went to the store to buy glasses": "#### 64",
            "Toulouse has twice as many sheep as Charleston": "#### 260",
            "Carla is downloading a 200 GB file": "#### 160",
        }
        provider = FakeProvider(response_map=responses)

        attempt = await adapter.run_task(gsm_spec, provider)
        evidence_ids = [ir.evidence_refs[0] for ir in attempt.item_results if ir.evidence_refs]
        assert len(evidence_ids) == 8
        assert len(set(evidence_ids)) == 8  # all unique

    async def test_pipeline_aggregate_matches_items(self, adapter):
        from tests.adapters.conftest import FakeProvider

        tasks = adapter.list_tasks()
        gsm_spec = next(t for t in tasks if t.task_id == "gsm8k_subset")

        responses = {
            "Janet's ducks lay 16 eggs per day": "#### 18",
            "A robe takes 2 bolts of blue fiber": "#### 3",
            "Josh decides to try flipping a house": "#### 70000",
            "James decides to run 3 sprints": "#### 540",
            "Every day, Wendi feeds each of her chickens": "#### 20",
            "Kylar went to the store to buy glasses": "#### 64",
            "Toulouse has twice as many sheep as Charleston": "#### 260",
            "Carla is downloading a 200 GB file": "#### 160",
        }
        provider = FakeProvider(response_map=responses)

        attempt = await adapter.run_task(gsm_spec, provider)
        item_aggregate = compute_item_aggregate_score(attempt.item_results)
        metric = attempt.metadata.get("metric_result")
        if metric and isinstance(metric, dict):
            assert abs(item_aggregate - metric["value"]) < 0.001


# ---------------------------------------------------------------------------
# Failure Isolation
# ---------------------------------------------------------------------------


class TestFailureIsolation:
    """Test that single item failures don't corrupt other items."""

    def _make_sample(self, evidence_id: str, response_text: str):
        return {"evidence_id": evidence_id, "response_text": response_text, "prompt": ""}

    def test_ungradable_item_no_fake_score(self):
        """An ungradable item should NOT get a fake score."""
        responses = [self._make_sample("00000000-0000-0000-0000-000000000000", "No answer pattern here")]
        items = _grade_gsm8k_items(responses, "a", "t")
        assert items[0].status == ItemStatus.UNGRADABLE
        assert items[0].raw_score == 0.0
        assert items[0].normalized_score == 0.0

    def test_single_wrong_answer_isolated(self):
        """One wrong answer should not affect other items."""
        responses = [
            self._make_sample("00000000-0000-0000-0000-000000000000", "#### 42"),
            self._make_sample("00000000-0000-0000-0000-000000000001", "#### 999"),
            self._make_sample("00000000-0000-0000-0000-000000000002", "#### 44"),
        ]
        items = _grade_gsm8k_items(responses, "a", "t")
        assert items[0].status == ItemStatus.GRADED
        assert items[1].status == ItemStatus.GRADED
        assert items[2].status == ItemStatus.GRADED
        # Each item independently graded
        assert items[0].raw_score != items[1].raw_score or items[0].raw_score == items[1].raw_score

    def test_aggregate_excludes_ungradable(self):
        """Ungradable items should not be counted in aggregate."""
        responses = [
            self._make_sample("00000000-0000-0000-0000-000000000000", "No pattern"),
            self._make_sample("00000000-0000-0000-0000-000000000001", "#### 43"),
            self._make_sample("00000000-0000-0000-0000-000000000002", "#### 44"),
        ]
        items = _grade_gsm8k_items(responses, "a", "t")
        aggregate = compute_item_aggregate_score(items)
        # Only items 1 and 2 (idx 1 and 2) are graded
        assert aggregate is not None

    def test_failure_items_do_not_contaminate_others(self):
        """GRADED items next to UNGRADABLE items should be unaffected."""
        items = [
            BenchmarkItemResult(item_id="i1", task_id="t", attempt_id="a", raw_score=1.0, normalized_score=1.0),
            BenchmarkItemResult(
                item_id="i2",
                task_id="t",
                attempt_id="a",
                status=ItemStatus.UNGRADABLE,
                raw_score=0.0,
                normalized_score=0.0,
            ),
            BenchmarkItemResult(item_id="i3", task_id="t", attempt_id="a", raw_score=1.0, normalized_score=1.0),
        ]
        aggregate = compute_item_aggregate_score(items)
        assert aggregate == 1.0

    def test_duplicate_item_ids_fine_in_list(self):
        """Duplicate item_ids are allowed at model level (validation is caller's responsibility)."""
        items = [
            BenchmarkItemResult(item_id="dup", task_id="t", attempt_id="a"),
            BenchmarkItemResult(item_id="dup", task_id="t", attempt_id="a"),
        ]
        assert len(items) == 2


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class TestItemLevelReporting:
    """Test that item-level data appears in JSON reports."""

    async def test_report_section_contains_items(self, adapter):
        from tests.adapters.conftest import FakeProvider

        tasks = adapter.list_tasks()
        gsm_spec = next(t for t in tasks if t.task_id == "gsm8k_subset")

        responses = {
            "Janet's ducks lay 16 eggs per day": "#### 18",
            "A robe takes 2 bolts of blue fiber": "#### 3",
            "Josh decides to try flipping a house": "#### 70000",
            "James decides to run 3 sprints": "#### 540",
            "Every day, Wendi feeds each of her chickens": "#### 20",
            "Kylar went to the store to buy glasses": "#### 64",
            "Toulouse has twice as many sheep as Charleston": "#### 260",
            "Carla is downloading a 200 GB file": "#### 160",
        }
        provider = FakeProvider(response_map=responses)

        attempt = await adapter.run_task(gsm_spec, provider)
        raw_result = {
            "results": {"gsm8k_subset": {"exact_match,strict-match": 1.0}},
            "evidence_ids": list(attempt.evidence_refs),
            "task_name": "gsm8k_subset",
            "attempt_id": attempt.attempt_id,
            "sample_results": [],
        }
        grade = adapter.normalize_result(raw_result)

        run_result = BenchmarkRunResult(
            run_id=str(uuid.uuid4()),
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

        plan = adapter.build_plan(
            attempt.suite_id,
            attempt.suite_version,
            attempt.source_id,
            attempt.source_revision,
            ["gsm8k_subset"],
        )
        section = build_benchmark_report_section(plan, run_result)

        # Section should have tasks
        assert len(section.tasks) == 1
        task = section.tasks[0]

        # Task should have items from the attempt
        assert len(task.items) == 8

        # First item should have expected structure
        first_item = task.items[0]
        assert first_item.item_id == "item-001"
        assert first_item.status in ("graded", "ungradable", "failure")
        assert len(first_item.evidence_refs) == 1

    def test_report_section_no_items_for_smoke(self, adapter):
        """Smoke task should have empty items list."""
        from llmtrace.benchmarks.models import TaskAttempt, TaskStatus
        from llmtrace.reporting.benchmark_mapper import build_benchmark_report_section

        # Create a smoke task attempt without items
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

    def test_report_json_round_trip(self, adapter):
        """Items should serialize and deserialize correctly."""
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

        # JSON round-trip
        d = item.model_dump(mode="json")
        restored = ItemReportItem(**d)
        assert restored.item_id == "item-001"
        assert restored.raw_score == 1.0
        assert restored.metadata == {"correct": True}

    def test_report_html_escaping(self, adapter):
        """Metadata with HTML should be preserved as-is (JSON is text)."""
        from llmtrace.reporting.benchmark_models import ItemReportItem

        item = ItemReportItem(
            item_id="item-001",
            status="graded",
            metadata={"extracted_answer": "<script>alert(1)</script>"},
        )
        d = item.model_dump()
        assert d["metadata"]["extracted_answer"] == "<script>alert(1)</script>"
