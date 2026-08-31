"""Tests for BehaviorSnapshotBuilder (behavior_snapshot.py)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest

from llmtrace.analysis.behavior_models import (
    DuplicateEvidenceError,
    DuplicateItemKeyError,
    ItemEvidenceError,
    MissingItemIdentityError,
)
from llmtrace.analysis.behavior_snapshot import BehaviorSnapshotBuilder
from llmtrace.benchmarks.models import (
    BenchmarkItemResult,
    BenchmarkRunResult,
    ItemStatus,
    TaskAttempt,
    TaskStatus,
)
from llmtrace.models.evidence import HTTPEvidence

from .conftest import (
    DEFAULT_ADAPTER_ID,
    DEFAULT_ADAPTER_VERSION,
    DEFAULT_POLICY_ID,
    DEFAULT_SOURCE_ID,
    DEFAULT_SOURCE_REVISION,
    DEFAULT_SUITE_ID,
    DEFAULT_SUITE_VERSION,
    _sha,
    make_evidence,
    make_profile,
    make_snapshot,
)

_BUILDER = BehaviorSnapshotBuilder()


def _make_run(
    items: list[BenchmarkItemResult],
    *,
    run_id: str = "run-x",
    task_id: str = "gsm8k_quick_v1",
) -> BenchmarkRunResult:
    all_refs = sorted({ref for item in items for ref in item.evidence_refs})
    attempt = TaskAttempt(
        attempt_id="att-1",
        task_id=task_id,
        status=TaskStatus.SUCCESS,
        source_id=DEFAULT_SOURCE_ID,
        source_revision=DEFAULT_SOURCE_REVISION,
        suite_id=DEFAULT_SUITE_ID,
        suite_version=DEFAULT_SUITE_VERSION,
        adapter_id=DEFAULT_ADAPTER_ID,
        adapter_version=DEFAULT_ADAPTER_VERSION,
        item_results=items,
        evidence_refs=all_refs,
    )
    return BenchmarkRunResult(
        run_id=run_id,
        task_attempts=[attempt],
        grade_results=[],
        source_id=DEFAULT_SOURCE_ID,
        source_revision=DEFAULT_SOURCE_REVISION,
        suite_id=DEFAULT_SUITE_ID,
        suite_version=DEFAULT_SUITE_VERSION,
        adapter_id=DEFAULT_ADAPTER_ID,
        adapter_version=DEFAULT_ADAPTER_VERSION,
        evidence_refs=all_refs,
    )


def _make_item(
    *,
    item_id: str,
    source_sample_id: str | None = "sample-1",
    input_sha256: str | None = None,
    status: ItemStatus = ItemStatus.GRADED,
    score: float = 1.0,
    evidence_refs: list[str] | None = None,
    task_id: str = "gsm8k_quick_v1",
) -> BenchmarkItemResult:
    return BenchmarkItemResult(
        item_id=item_id,
        task_id=task_id,
        attempt_id="att-1",
        source_sample_id=source_sample_id,
        input_sha256=input_sha256 if input_sha256 is not None else _sha(source_sample_id or item_id),
        status=status,
        raw_score=score,
        normalized_score=score,
        evidence_refs=evidence_refs if evidence_refs is not None else [],
    )


# ===========================================================================
# Happy path
# ===========================================================================


class TestBuilderHappyPath:
    def test_32_items(self) -> None:
        items = [
            {
                "task_id": f"task-{i % 4}",
                "source_sample_id": f"sample-{i}",
                "status": ItemStatus.GRADED,
                "score": 1.0,
                "response_text": f"answer {i}",
            }
            for i in range(32)
        ]
        snap = make_snapshot(items=items)
        assert len(snap.items) == 32
        assert snap.suite_id == DEFAULT_SUITE_ID
        assert snap.adapter_id == DEFAULT_ADAPTER_ID
        assert snap.scoring_policy_id == DEFAULT_POLICY_ID

    def test_item_order_is_deterministic(self) -> None:
        items_a = [
            {"task_id": "t1", "source_sample_id": "b", "status": ItemStatus.GRADED, "score": 1.0},
            {"task_id": "t1", "source_sample_id": "a", "status": ItemStatus.GRADED, "score": 1.0},
            {"task_id": "t2", "source_sample_id": "c", "status": ItemStatus.GRADED, "score": 1.0},
        ]
        items_b = [
            {"task_id": "t2", "source_sample_id": "c", "status": ItemStatus.GRADED, "score": 1.0},
            {"task_id": "t1", "source_sample_id": "a", "status": ItemStatus.GRADED, "score": 1.0},
            {"task_id": "t1", "source_sample_id": "b", "status": ItemStatus.GRADED, "score": 1.0},
        ]
        snap_a = make_snapshot(items=items_a)
        snap_b = make_snapshot(items=items_b)
        keys_a = [o.key.sort_key for o in snap_a.items]
        keys_b = [o.key.sort_key for o in snap_b.items]
        assert keys_a == keys_b
        assert keys_a == sorted(keys_a)

    def test_evidence_union_is_deduped_first_seen(self) -> None:
        ev = make_evidence()
        item = _make_item(item_id="i1", source_sample_id="s1", evidence_refs=[str(ev.evidence_id)])
        run = _make_run([item])
        snap = _BUILDER.build(
            run_results=[run],
            profile=make_profile(),
            evidence=[ev],
            target_id="t",
            candidate_model_id="m",
            generation_config={"temperature": 0.0},
        )
        assert snap.evidence_refs == (str(ev.evidence_id),)

    def test_full_response_text_not_saved(self) -> None:
        secret_text = "The answer is 42 and it is definitely correct."
        snap = make_snapshot(
            items=[
                {
                    "task_id": "t1",
                    "source_sample_id": "s1",
                    "status": ItemStatus.GRADED,
                    "score": 1.0,
                    "response_text": secret_text,
                }
            ]
        )
        dumped = snap.model_dump(mode="json")
        serialized = repr(dumped)
        assert secret_text not in serialized
        obs = snap.items[0]
        assert obs.output_length == len(secret_text)
        assert not hasattr(obs, "output_text")

    def test_created_at_is_utc_aware(self) -> None:
        snap = make_snapshot()
        assert snap.created_at.tzinfo is not None
        assert snap.created_at.utcoffset() is not None


# ===========================================================================
# Fail-closed builder invariants
# ===========================================================================


class TestBuilderFailClosed:
    def test_missing_source_sample_id_rejected(self) -> None:
        ev = make_evidence()
        item = _make_item(item_id="i1", source_sample_id=None, evidence_refs=[str(ev.evidence_id)])
        run = _make_run([item])
        with pytest.raises(MissingItemIdentityError):
            _BUILDER.build(
                run_results=[run],
                profile=make_profile(),
                evidence=[ev],
                target_id="t",
                candidate_model_id="m",
                generation_config={"temperature": 0.0},
            )

    def test_missing_input_sha256_rejected(self) -> None:
        ev = make_evidence()
        item = BenchmarkItemResult(
            item_id="i1",
            task_id="gsm8k_quick_v1",
            attempt_id="att-1",
            source_sample_id="s1",
            input_sha256=None,
            status=ItemStatus.GRADED,
            raw_score=1.0,
            normalized_score=1.0,
            evidence_refs=[str(ev.evidence_id)],
        )
        run = _make_run([item])
        with pytest.raises(MissingItemIdentityError):
            _BUILDER.build(
                run_results=[run],
                profile=make_profile(),
                evidence=[ev],
                target_id="t",
                candidate_model_id="m",
                generation_config={"temperature": 0.0},
            )

    def test_duplicate_stable_key_rejected(self) -> None:
        ev1 = make_evidence(response_text="one")
        ev2 = make_evidence(response_text="two")
        item1 = _make_item(item_id="i1", source_sample_id="same", evidence_refs=[str(ev1.evidence_id)])
        item2 = _make_item(item_id="i2", source_sample_id="same", evidence_refs=[str(ev2.evidence_id)])
        run = _make_run([item1, item2])
        with pytest.raises(DuplicateItemKeyError):
            _BUILDER.build(
                run_results=[run],
                profile=make_profile(),
                evidence=[ev1, ev2],
                target_id="t",
                candidate_model_id="m",
                generation_config={"temperature": 0.0},
            )

    def test_unknown_evidence_rejected(self) -> None:
        # The item references a valid UUID that the run claims, but the builder
        # is not given that evidence object — so the reference is unresolvable.
        unknown_ref = str(uuid4())
        ev = make_evidence()
        item = _make_item(item_id="i1", source_sample_id="s1", evidence_refs=[unknown_ref])
        run = _make_run([item])
        with pytest.raises(ItemEvidenceError):
            _BUILDER.build(
                run_results=[run],
                profile=make_profile(),
                evidence=[ev],  # does not contain `unknown_ref`
                target_id="t",
                candidate_model_id="m",
                generation_config={"temperature": 0.0},
            )

    def test_multiple_evidence_ambiguity_fails_closed(self) -> None:
        ev1 = make_evidence(response_text="one")
        ev2 = make_evidence(response_text="two")
        item = _make_item(
            item_id="i1",
            source_sample_id="s1",
            evidence_refs=[str(ev1.evidence_id), str(ev2.evidence_id)],
        )
        run = _make_run([item])
        with pytest.raises(ItemEvidenceError):
            _BUILDER.build(
                run_results=[run],
                profile=make_profile(),
                evidence=[ev1, ev2],
                target_id="t",
                candidate_model_id="m",
                generation_config={"temperature": 0.0},
            )

    def test_empty_run_results_rejected(self) -> None:
        with pytest.raises(ValueError):
            _BUILDER.build(
                run_results=[],
                profile=make_profile(),
                evidence=[],
                target_id="t",
                candidate_model_id="m",
                generation_config={"temperature": 0.0},
            )

    def test_mixed_suite_context_rejected(self) -> None:
        ev = make_evidence()
        item = _make_item(item_id="i1", source_sample_id="s1", evidence_refs=[str(ev.evidence_id)])
        run1 = _make_run([item], run_id="run-1")
        run2 = _make_run([item], run_id="run-2")
        run2 = run2.model_copy(update={"suite_id": "other_suite"})
        with pytest.raises(ValueError):
            _BUILDER.build(
                run_results=[run1, run2],
                profile=make_profile(),
                evidence=[ev],
                target_id="t",
                candidate_model_id="m",
                generation_config={"temperature": 0.0},
            )


class TestBuilderOperationalExtraction:
    def test_operational_fields_extracted(self) -> None:
        ev = make_evidence(
            response_text="42",
            response_model="gpt-4o",
            finish_reason="length",
            input_tokens=7,
            output_tokens=3,
            latency_ms=321.0,
        )
        item = _make_item(item_id="i1", source_sample_id="s1", evidence_refs=[str(ev.evidence_id)])
        run = _make_run([item])
        snap = _BUILDER.build(
            run_results=[run],
            profile=make_profile(),
            evidence=[ev],
            target_id="t",
            candidate_model_id="m",
            generation_config={"temperature": 0.0},
        )
        obs = snap.items[0]
        assert obs.response_model == "gpt-4o"
        assert obs.finish_reason == "length"
        assert obs.input_tokens == 7
        assert obs.output_tokens == 3
        assert obs.latency_ms == 321.0
        assert obs.output_length == 2


class TestBuilderMultiRun:
    def test_multi_run_id_is_joined_deterministically(self) -> None:
        ev1 = make_evidence(response_text="one")
        ev2 = make_evidence(response_text="two")
        item1 = _make_item(item_id="i1", source_sample_id="s1", evidence_refs=[str(ev1.evidence_id)])
        item2 = _make_item(item_id="i2", source_sample_id="s2", evidence_refs=[str(ev2.evidence_id)])
        run1 = _make_run([item1], run_id="run-1")
        run2 = _make_run([item2], run_id="run-2")
        snap = _BUILDER.build(
            run_results=[run1, run2],
            profile=make_profile(),
            evidence=[ev1, ev2],
            target_id="t",
            candidate_model_id="m",
            generation_config={"temperature": 0.0},
        )
        assert snap.run_id == "run-1::run-2"
        assert len(snap.items) == 2

    def test_created_at_falls_back_to_utc_now(self) -> None:
        ev = make_evidence()
        item = _make_item(item_id="i1", source_sample_id="s1", evidence_refs=[str(ev.evidence_id)])
        run = _make_run([item])  # no finished_at set
        snap = _BUILDER.build(
            run_results=[run],
            profile=make_profile(),
            evidence=[ev],
            target_id="t",
            candidate_model_id="m",
            generation_config={"temperature": 0.0},
        )
        assert snap.created_at.tzinfo is not None
        assert snap.created_at.utcoffset() is not None


class TestBuilderTimestampSafety:
    def _run_with_finished_at(self, finished_at: datetime | None) -> tuple[BenchmarkRunResult, HTTPEvidence]:
        ev = make_evidence()
        item = _make_item(item_id="i1", source_sample_id="s1", evidence_refs=[str(ev.evidence_id)])
        run = _make_run([item]).model_copy(update={"finished_at": finished_at})
        return run, ev

    def test_aware_finished_at_normalized_to_utc(self) -> None:
        plus_eight = timezone(timedelta(hours=8))
        run, ev = self._run_with_finished_at(datetime(2026, 8, 31, 8, 0, tzinfo=plus_eight))
        snap = _BUILDER.build(
            run_results=[run],
            profile=make_profile(),
            evidence=[ev],
            target_id="t",
            candidate_model_id="m",
            generation_config={"temperature": 0.0},
        )
        assert snap.created_at.tzinfo is UTC
        assert snap.created_at == datetime(2026, 8, 31, 0, 0, tzinfo=UTC)

    def test_naive_finished_at_rejected(self) -> None:
        run, ev = self._run_with_finished_at(datetime(2026, 8, 31, 8, 0, 0))
        with pytest.raises(ValueError, match="timezone-aware"):
            _BUILDER.build(
                run_results=[run],
                profile=make_profile(),
                evidence=[ev],
                target_id="t",
                candidate_model_id="m",
                generation_config={"temperature": 0.0},
            )


class TestBuilderDuplicateEvidence:
    def test_duplicate_evidence_id_fails_closed(self) -> None:
        eid = uuid4()
        ev1 = HTTPEvidence(
            evidence_id=eid,
            request_method="POST",
            request_url_redacted="https://api.example.com",
            request_path="/",
            request_headers_redacted={},
            response_text="one",
        )
        ev2 = HTTPEvidence(
            evidence_id=eid,
            request_method="POST",
            request_url_redacted="https://api.example.com",
            request_path="/",
            request_headers_redacted={},
            response_text="two",
        )
        item = _make_item(item_id="i1", source_sample_id="s1", evidence_refs=[str(eid)])
        run = _make_run([item])
        with pytest.raises(DuplicateEvidenceError):
            _BUILDER.build(
                run_results=[run],
                profile=make_profile(),
                evidence=[ev1, ev2],
                target_id="t",
                candidate_model_id="m",
                generation_config={"temperature": 0.0},
            )
