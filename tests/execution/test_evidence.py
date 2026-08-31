"""Tests for the central EvidenceRecorder (execution/evidence.py)."""

from __future__ import annotations

import pytest

from llmtrace.analysis.behavior_models import DuplicateEvidenceError
from llmtrace.execution.evidence import InMemoryEvidenceRecorder

from .conftest import make_evidence


class TestInMemoryEvidenceRecorder:
    def test_records_in_arrival_order(self) -> None:
        rec = InMemoryEvidenceRecorder()
        a = make_evidence()
        b = make_evidence()
        c = make_evidence()
        rec.record(a)
        rec.record(b)
        rec.record(c)
        assert [e.evidence_id for e in rec.list()] == [a.evidence_id, b.evidence_id, c.evidence_id]

    def test_duplicate_evidence_id_fails_closed(self) -> None:
        rec = InMemoryEvidenceRecorder()
        a = make_evidence()
        rec.record(a)
        dup = make_evidence()
        dup.evidence_id = a.evidence_id  # type: ignore[misc]
        with pytest.raises(DuplicateEvidenceError):
            rec.record(dup)
        # The duplicate was NOT appended — exactly-once is preserved.
        assert len(rec) == 1

    def test_snapshot_is_independent_copy(self) -> None:
        rec = InMemoryEvidenceRecorder()
        rec.record(make_evidence())
        snap = rec.snapshot()
        assert len(snap) == 1
        # Recording more after snapshot() must not mutate the returned tuple.
        rec.record(make_evidence())
        assert len(snap) == 1
        assert len(rec) == 2

    def test_does_not_mutate_recorded_evidence(self) -> None:
        rec = InMemoryEvidenceRecorder()
        ev = make_evidence()
        original_text = ev.response_text
        rec.record(ev)
        assert rec.list()[0].response_text == original_text

    def test_contains_and_len(self) -> None:
        rec = InMemoryEvidenceRecorder()
        assert len(rec) == 0
        ev = make_evidence()
        rec.record(ev)
        assert len(rec) == 1
        assert str(ev.evidence_id) in rec
        assert "does-not-exist" not in rec

    def test_empty_recorder(self) -> None:
        rec = InMemoryEvidenceRecorder()
        assert rec.list() == ()
        assert rec.snapshot() == ()
