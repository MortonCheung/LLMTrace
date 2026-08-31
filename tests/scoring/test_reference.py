"""Tests for ReferenceSnapshot, ReferenceProvenance, and ReferenceRepository."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from llmtrace.scoring.errors import DuplicateSnapshotError, ReferenceNotFoundError
from llmtrace.scoring.models import (
    CapabilityDimension,
    CapabilityProfile,
    DimensionScoreResult,
    DimensionScoreStatus,
)
from llmtrace.scoring.reference import (
    ReferenceProvenance,
    ReferenceRepository,
    ReferenceSnapshot,
)

_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "reference_profiles"

_FULL_SCORES: dict[CapabilityDimension, float] = {
    CapabilityDimension.REASONING: 1.0,
    CapabilityDimension.CODING: 1.0,
    CapabilityDimension.MATH_SCIENCE: 1.0,
    CapabilityDimension.INSTRUCTION_FOLLOWING: 1.0,
}


def _dim(dimension: CapabilityDimension, score: float) -> DimensionScoreResult:
    return DimensionScoreResult(
        dimension=dimension,
        status=DimensionScoreStatus.UNCALIBRATED,
        raw_normalized_score=score,
    )


def _profile(scores: dict[CapabilityDimension, float], *, coverage_weight: float = 0.75) -> CapabilityProfile:
    return CapabilityProfile(
        scoring_policy_id="llmtrace-capability-v1",
        scoring_policy_version="0.1.0",
        dimensions=tuple(_dim(d, s) for d, s in scores.items()),
        coverage_weight=coverage_weight,
    )


def _provenance() -> ReferenceProvenance:
    return ReferenceProvenance(
        source_type="benchmark_run",
        created_by="llmtrace",
        created_at=datetime(2026, 8, 10, tzinfo=UTC),
        suite_sha256="abc123",
        benchmark_revision="quick-v1-rev",
        runner_version="0.3.0",
    )


def _snapshot(
    *,
    snapshot_id: str = "openai-gpt-x-quick-v1",
    model_id: str = "gpt-x",
    provider_id: str = "openai",
    suite_id: str = "llmtrace_quick_v1",
    suite_version: str = "0.1.0",
    profile: CapabilityProfile | None = None,
) -> ReferenceSnapshot:
    return ReferenceSnapshot(
        snapshot_id=snapshot_id,
        model_id=model_id,
        provider_id=provider_id,
        created_at=datetime(2026, 8, 10, tzinfo=UTC),
        suite_id=suite_id,
        suite_version=suite_version,
        capability_profile=profile if profile is not None else _profile(_FULL_SCORES),
        provenance=_provenance(),
    )


class TestReferenceSnapshotSchema:
    def test_create_snapshot(self) -> None:
        snap = _snapshot()
        assert snap.snapshot_id == "openai-gpt-x-quick-v1"
        assert snap.model_id == "gpt-x"
        assert snap.provider_id == "openai"
        assert snap.suite_id == "llmtrace_quick_v1"
        assert snap.suite_version == "0.1.0"
        assert snap.capability_profile.scoring_policy_id == "llmtrace-capability-v1"

    def test_provenance_fields_present(self) -> None:
        prov = _provenance()
        assert prov.source_type == "benchmark_run"
        assert prov.created_by == "llmtrace"
        assert prov.suite_sha256 == "abc123"
        assert prov.benchmark_revision == "quick-v1-rev"
        assert prov.runner_version == "0.3.0"

    def test_serialize_deserialize_roundtrip(self) -> None:
        snap = _snapshot()
        data = snap.model_dump(mode="json")
        restored = ReferenceSnapshot.model_validate(data)
        assert restored == snap
        assert restored.capability_profile.dimensions == snap.capability_profile.dimensions
        assert restored.provenance.created_at == snap.provenance.created_at

    def test_empty_snapshot_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            _snapshot(snapshot_id="")


class TestReferenceSnapshotImmutable:
    def test_snapshot_is_frozen(self) -> None:
        snap = _snapshot()
        with pytest.raises((TypeError, ValueError)):
            snap.model_id = "other"  # type: ignore[misc]

    def test_provenance_is_frozen(self) -> None:
        prov = _provenance()
        with pytest.raises((TypeError, ValueError)):
            prov.source_type = "manual"  # type: ignore[misc]

    def test_repository_save_rejects_duplicate(self) -> None:
        repo = ReferenceRepository()
        repo.save(_snapshot())
        with pytest.raises(DuplicateSnapshotError):
            repo.save(_snapshot())

    def test_no_update_snapshot_api(self) -> None:
        """No mutation method exists on ReferenceSnapshot or Repository."""
        assert not hasattr(ReferenceSnapshot, "update_snapshot")
        assert not hasattr(ReferenceRepository, "update_snapshot")


class TestReferenceRepository:
    def test_save_and_get(self) -> None:
        repo = ReferenceRepository()
        snap = _snapshot()
        repo.save(snap)
        assert repo.get("openai-gpt-x-quick-v1") == snap

    def test_save_writes_json_file(self, tmp_path: Path) -> None:
        repo = ReferenceRepository(directory=tmp_path)
        repo.save(_snapshot())
        path = tmp_path / "openai-gpt-x-quick-v1.json"
        assert path.exists()
        data = path.read_text(encoding="utf-8")
        assert '"snapshot_id": "openai-gpt-x-quick-v1"' in data

    def test_get_missing_raises(self) -> None:
        repo = ReferenceRepository()
        with pytest.raises(ReferenceNotFoundError):
            repo.get("does-not-exist")

    def test_list_is_sorted(self) -> None:
        repo = ReferenceRepository()
        repo.save(_snapshot(snapshot_id="b-snap", model_id="model-b"))
        repo.save(_snapshot(snapshot_id="a-snap", model_id="model-a"))
        ids = [s.snapshot_id for s in repo.list()]
        assert ids == ["a-snap", "b-snap"]

    def test_find_by_model(self) -> None:
        repo = ReferenceRepository()
        repo.save(_snapshot(snapshot_id="gpt-v1", model_id="gpt-x"))
        repo.save(_snapshot(snapshot_id="gpt-v2", model_id="gpt-x"))
        repo.save(_snapshot(snapshot_id="claude-v1", model_id="claude-x"))
        found = repo.find_by_model("gpt-x")
        assert {s.snapshot_id for s in found} == {"gpt-v1", "gpt-v2"}
        assert repo.find_by_model("unknown") == []

    def test_len_and_contains(self) -> None:
        repo = ReferenceRepository()
        assert len(repo) == 0
        repo.save(_snapshot())
        assert len(repo) == 1
        assert "openai-gpt-x-quick-v1" in repo
        assert "missing" not in repo

    def test_load_from_fixture_directory(self) -> None:
        repo = ReferenceRepository.load(_FIXTURE_DIR)
        snap = repo.get("openai-gpt-x-quick-v1")
        assert snap.model_id == "gpt-x"
        assert snap.provider_id == "openai"
        assert snap.suite_id == "llmtrace_quick_v1"
        assert len(snap.capability_profile.dimensions) == 4

    def test_save_then_load_roundtrip(self, tmp_path: Path) -> None:
        repo = ReferenceRepository(directory=tmp_path)
        repo.save(_snapshot())
        loaded = ReferenceRepository.load(tmp_path)
        assert loaded.get("openai-gpt-x-quick-v1") == _snapshot()
