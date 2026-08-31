"""Tests for ReferenceSnapshot, ReferenceProvenance, and ReferenceRepository."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
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

# A real 64-character hex SHA-256 digest (placeholder content, valid format).
_TEST_SUITE_SHA256 = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
_TEST_SUITE_SHA256_UPPER = _TEST_SUITE_SHA256.upper()

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


def _provenance(
    *,
    suite_sha256: str = _TEST_SUITE_SHA256,
    created_at: datetime | None = None,
) -> ReferenceProvenance:
    return ReferenceProvenance(
        source_type="benchmark_run",
        created_by="llmtrace",
        created_at=created_at if created_at is not None else datetime(2026, 8, 10, tzinfo=UTC),
        suite_sha256=suite_sha256,
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
    created_at: datetime | None = None,
    provenance: ReferenceProvenance | None = None,
) -> ReferenceSnapshot:
    return ReferenceSnapshot(
        snapshot_id=snapshot_id,
        model_id=model_id,
        provider_id=provider_id,
        created_at=created_at if created_at is not None else datetime(2026, 8, 10, tzinfo=UTC),
        suite_id=suite_id,
        suite_version=suite_version,
        capability_profile=profile if profile is not None else _profile(_FULL_SCORES),
        provenance=provenance if provenance is not None else _provenance(),
    )


# ===========================================================================
# Schema
# ===========================================================================


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
        assert prov.suite_sha256 == _TEST_SUITE_SHA256
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


# ===========================================================================
# Provenance validation — suite_sha256 must be a real SHA-256
# ===========================================================================


class TestSuiteSha256Validation:
    def test_valid_64_hex_accepted(self) -> None:
        prov = _provenance(suite_sha256=_TEST_SUITE_SHA256)
        assert prov.suite_sha256 == _TEST_SUITE_SHA256

    def test_uppercase_hex_normalized_to_lowercase(self) -> None:
        prov = _provenance(suite_sha256=_TEST_SUITE_SHA256_UPPER)
        assert prov.suite_sha256 == _TEST_SUITE_SHA256
        assert prov.suite_sha256 == prov.suite_sha256.lower()

    @pytest.mark.parametrize(
        "bad",
        [
            "abc123",  # too short placeholder — the exact value this fix removes
            "",  # empty
            "z" * 64,  # non-hex characters
            "a" * 63,  # one char short
            "a" * 65,  # one char long
            "a" * 63 + " ",  # trailing space
        ],
    )
    def test_invalid_sha256_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError):
            _provenance(suite_sha256=bad)

    def test_error_message_states_sha256_requirement(self) -> None:
        with pytest.raises(ValueError, match="64 hexadecimal characters"):
            _provenance(suite_sha256="abc123")


# ===========================================================================
# Provenance validation — created_at must be timezone-aware UTC
# ===========================================================================


class TestCreatedAtTimezoneValidation:
    @pytest.mark.parametrize(
        "naive",
        [
            datetime(2026, 8, 31),
            datetime(2026, 8, 31, 12, 0, 0),
            datetime(2026, 8, 31, 12, 0, 0, 500000),
        ],
    )
    def test_naive_datetime_rejected_in_provenance(self, naive: datetime) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            _provenance(created_at=naive)

    def test_naive_datetime_rejected_in_snapshot(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            _snapshot(created_at=datetime(2026, 8, 31))

    def test_utc_datetime_accepted(self) -> None:
        snap = _snapshot(created_at=datetime(2026, 8, 31, tzinfo=UTC))
        assert snap.created_at.tzinfo is UTC

    def test_non_utc_aware_datetime_normalized_to_utc(self) -> None:
        plus_eight = timezone(timedelta(hours=8))
        prov = _provenance(created_at=datetime(2026, 8, 10, 8, 0, tzinfo=plus_eight))
        assert prov.created_at.tzinfo is UTC
        assert prov.created_at == datetime(2026, 8, 10, 0, 0, tzinfo=UTC)

    def test_snapshot_created_at_normalized_to_utc(self) -> None:
        plus_eight = timezone(timedelta(hours=8))
        snap = _snapshot(created_at=datetime(2026, 8, 10, 8, 0, tzinfo=plus_eight))
        assert snap.created_at.tzinfo is UTC
        assert snap.created_at == datetime(2026, 8, 10, 0, 0, tzinfo=UTC)

    def test_normalized_utc_survives_json_roundtrip(self) -> None:
        plus_eight = timezone(timedelta(hours=8))
        snap = _snapshot(created_at=datetime(2026, 8, 10, 8, 0, tzinfo=plus_eight))
        restored = ReferenceSnapshot.model_validate(snap.model_dump(mode="json"))
        assert restored.created_at == datetime(2026, 8, 10, 0, 0, tzinfo=UTC)


# ===========================================================================
# snapshot_id must be filename-safe
# ===========================================================================


class TestSnapshotIdSafety:
    @pytest.mark.parametrize(
        "unsafe",
        [
            "../../evil",
            "../evil",
            "..",
            ".",
            "foo/bar",
            "foo\\bar",
            "/absolute",
            " leading-space",
            "trailing-space ",
            "with space",
            "..%2Fevil",
        ],
    )
    def test_unsafe_snapshot_id_rejected(self, unsafe: str) -> None:
        with pytest.raises(ValueError, match="filename-safe"):
            _snapshot(snapshot_id=unsafe)

    @pytest.mark.parametrize(
        "safe",
        [
            "openai-gpt-x-20260831-quick-v1",
            "claude-sonnet-4.5-quick-v1",
            "reference_v2",
            "snap-1",
        ],
    )
    def test_safe_snapshot_id_accepted(self, safe: str) -> None:
        assert _snapshot(snapshot_id=safe).snapshot_id == safe

    @pytest.mark.parametrize("escaping", ["../../evil", "../evil", "foo/bar", "/absolute", "../../../deep/evil"])
    def test_file_path_rejects_escaping_repository_directory(self, tmp_path: Path, escaping: str) -> None:
        repo_dir = tmp_path / "repo"
        repo = ReferenceRepository(directory=repo_dir)
        with pytest.raises(ValueError, match="outside the repository directory"):
            repo._file_path(escaping)

    def test_no_json_file_created_outside_repository_directory(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        repo = ReferenceRepository(directory=repo_dir)

        with pytest.raises(ValueError):
            repo._file_path("../../evil")

        assert not list(tmp_path.glob("*.json"))
        assert not list(tmp_path.parent.glob("evil.json"))
        assert not (tmp_path.parent / "evil.json").exists()

    def test_save_refuses_traversal_even_if_model_validation_bypassed(self, tmp_path: Path) -> None:
        """Defence in depth: the repository must not trust snapshot_id blindly."""
        repo_dir = tmp_path / "repo"
        repo = ReferenceRepository(directory=repo_dir)

        bypassed = ReferenceSnapshot.model_construct(
            snapshot_id="../../evil",
            model_id="evil",
            provider_id="evil",
            created_at=datetime(2026, 8, 10, tzinfo=UTC),
            suite_id="llmtrace_quick_v1",
            suite_version="0.1.0",
            capability_profile=_profile(_FULL_SCORES),
            provenance=_provenance(),
        )

        with pytest.raises(ValueError, match="outside the repository directory"):
            repo.save(bypassed)

        assert len(repo) == 0
        assert not (tmp_path / "evil.json").exists()
        assert not (tmp_path.parent / "evil.json").exists()


# ===========================================================================
# Immutability
# ===========================================================================


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


# ===========================================================================
# Repository — append-only on disk, not just in memory
# ===========================================================================


class TestRepositoryAppendOnly:
    def test_existing_disk_file_cannot_be_overwritten(self, tmp_path: Path) -> None:
        """The highest-priority v0.3-C invariant: history on disk is untouchable."""
        target = tmp_path / "openai-gpt-x-quick-v1.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        original_content = '{"pre-existing": "historical record"}'
        target.write_text(original_content, encoding="utf-8")

        repo = ReferenceRepository(directory=tmp_path)
        with pytest.raises(DuplicateSnapshotError):
            repo.save(_snapshot())

        assert target.read_text(encoding="utf-8") == original_content

    def test_fresh_repository_cannot_overwrite_previous_snapshot(self, tmp_path: Path) -> None:
        repo1 = ReferenceRepository(directory=tmp_path)
        repo1.save(_snapshot())

        path = tmp_path / "openai-gpt-x-quick-v1.json"
        content_after_first_save = path.read_text(encoding="utf-8")

        repo2 = ReferenceRepository(directory=tmp_path)
        with pytest.raises(DuplicateSnapshotError):
            repo2.save(_snapshot())

        assert path.read_text(encoding="utf-8") == content_after_first_save

    def test_repository_loaded_from_disk_also_refuses_duplicate(self, tmp_path: Path) -> None:
        ReferenceRepository(directory=tmp_path).save(_snapshot())

        loaded = ReferenceRepository.load(tmp_path)
        with pytest.raises(DuplicateSnapshotError):
            loaded.save(_snapshot(snapshot_id="openai-gpt-x-quick-v1"))

    def test_new_version_requires_new_snapshot_id(self, tmp_path: Path) -> None:
        repo = ReferenceRepository(directory=tmp_path)
        repo.save(_snapshot(snapshot_id="gpt-x-2026-08-10"))
        repo.save(_snapshot(snapshot_id="gpt-x-2026-08-31"))
        assert len(repo) == 2
        assert (tmp_path / "gpt-x-2026-08-10.json").exists()
        assert (tmp_path / "gpt-x-2026-08-31.json").exists()

    def test_disk_write_failure_does_not_contaminate_memory_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _simulated_disk_failure(*args: object, **kwargs: object) -> None:
            raise OSError("simulated disk full")

        monkeypatch.setattr(Path, "open", _simulated_disk_failure)

        repo = ReferenceRepository(directory=tmp_path)
        with pytest.raises(OSError, match="simulated disk full"):
            repo.save(_snapshot())

        assert "openai-gpt-x-quick-v1" not in repo
        assert len(repo) == 0

    def test_disk_write_failure_leaves_repository_usable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _simulated_disk_failure(*args: object, **kwargs: object) -> None:
            raise OSError("simulated disk full")

        repo = ReferenceRepository(directory=tmp_path)
        monkeypatch.setattr(Path, "open", _simulated_disk_failure)
        with pytest.raises(OSError):
            repo.save(_snapshot())

        monkeypatch.undo()
        repo.save(_snapshot(snapshot_id="recovered-snapshot"))
        assert repo.get("recovered-snapshot").snapshot_id == "recovered-snapshot"

    def test_loading_duplicate_snapshot_ids_from_directory_rejected(self, tmp_path: Path) -> None:
        """Same snapshot_id under two different filenames is still a duplicate."""
        repo = ReferenceRepository(directory=tmp_path)
        repo.save(_snapshot(snapshot_id="dup-snap"))
        (tmp_path / "copy-of-dup-snap.json").write_text(
            (tmp_path / "dup-snap.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        with pytest.raises(DuplicateSnapshotError, match="Duplicate ReferenceSnapshot"):
            ReferenceRepository.load(tmp_path)


# ===========================================================================
# Repository — basic operations
# ===========================================================================


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


# ===========================================================================
# Fixture honesty
# ===========================================================================


class TestFixtureIsNotARealModelResult:
    def test_fixture_provenance_marks_itself_as_test_data(self) -> None:
        snap = ReferenceRepository.load(_FIXTURE_DIR).get("openai-gpt-x-quick-v1")
        assert snap.provenance.source_type == "test_fixture"
        assert "fixture" in snap.provenance.created_by

    def test_fixture_suite_sha256_is_valid_sha256(self) -> None:
        snap = ReferenceRepository.load(_FIXTURE_DIR).get("openai-gpt-x-quick-v1")
        assert len(snap.provenance.suite_sha256) == 64
        assert snap.provenance.suite_sha256 == snap.provenance.suite_sha256.lower()
        int(snap.provenance.suite_sha256, 16)

    def test_fixture_calibrated_scores_stay_none(self) -> None:
        snap = ReferenceRepository.load(_FIXTURE_DIR).get("openai-gpt-x-quick-v1")
        assert snap.capability_profile.calibrated_total_score is None
        assert all(d.calibrated_score is None for d in snap.capability_profile.dimensions)
