"""Unit tests for FakeProvider evidence storage."""

from __future__ import annotations

import pytest

from tests.adapters.conftest import (
    FakeProvider,
    FakeProviderError,
)


class TestFakeProviderEvidenceStorage:
    """Evidence stored in self.evidence list."""

    def test_initial_evidence_empty(self) -> None:
        p = FakeProvider(response_map={"hello": "world"})
        assert p.evidence == []

    @pytest.mark.asyncio
    async def test_single_complete_stores_one_evidence(self) -> None:
        p = FakeProvider(response_map={"hello": "DETERMINISTIC"})
        ev = await p.complete(
            model="test",
            messages=[{"role": "user", "content": "hello world"}],
        )
        assert len(p.evidence) == 1
        assert p.evidence[0] is ev

    @pytest.mark.asyncio
    async def test_four_completes_store_ordered_four(self) -> None:
        p = FakeProvider(
            response_map={
                "a": "ra",
                "b": "rb",
                "c": "rc",
                "d": "rd",
            }
        )
        evs = []
        for ch in ["a", "b", "c", "d"]:
            ev = await p.complete(model="m", messages=[{"role": "user", "content": ch}])
            evs.append(ev)
        assert len(p.evidence) == 4
        for i in range(4):
            assert p.evidence[i] is evs[i]

    @pytest.mark.asyncio
    async def test_evidence_ids_unique(self) -> None:
        p = FakeProvider(response_map={"x": "rx"})
        ids = set()
        for _ in range(4):
            ev = await p.complete(model="m", messages=[{"role": "user", "content": "x"}])
            ids.add(str(ev.evidence_id))
        assert len(ids) == 4

    @pytest.mark.asyncio
    async def test_failure_does_not_save_evidence(self) -> None:
        p = FakeProvider(fail_on_call=1, fail_error="boom")
        with pytest.raises(FakeProviderError, match="boom"):
            await p.complete(model="m", messages=[{"role": "user", "content": "test"}])
        assert p.evidence == []

    @pytest.mark.asyncio
    async def test_exception_evidence_still_saved(self) -> None:
        """Evidence with exception_type is still a valid return — must be saved."""
        p = FakeProvider(
            fail_with_exception_type="ConnectionError",
            fail_with_exception_message="Simulated error",
        )
        ev = await p.complete(model="m", messages=[{"role": "user", "content": "test"}])
        assert len(p.evidence) == 1
        assert p.evidence[0] is ev
        assert ev.exception_type == "ConnectionError"
