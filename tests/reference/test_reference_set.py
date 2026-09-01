"""ReferenceSet / ReferenceSetRepository tests (§41): 12-gate compatibility, immutability."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from llmtrace.reference.reference_set import (
    DuplicateReferenceSetError,
    FixtureReferenceError,
    ReferenceSet,
    ReferenceSetBuilder,
    ReferenceSetCompatibilityError,
    ReferenceSetError,
    ReferenceSetIntegrityError,
    ReferenceSetNotFoundError,
)
from llmtrace.reference.repository import ReferenceSetRepository
from llmtrace.scoring.models import (
    CapabilityDimension,
    CapabilityProfile,
    DimensionScoreResult,
    DimensionScoreStatus,
)
from llmtrace.scoring.reference import ReferenceSnapshot

from .helpers import make_snapshot


def _snapshot(
    snapshot_id: str = "snap-a",
    *,
    model_id: str = "gpt-x",
    provenance_update: dict[str, object] | None = None,
    profile_update: dict[str, object] | None = None,
) -> ReferenceSnapshot:
    snapshot = make_snapshot(snapshot_id=snapshot_id, model_id=model_id)
    if provenance_update:
        snapshot = snapshot.model_copy(update={"provenance": snapshot.provenance.model_copy(update=provenance_update)})
    if profile_update:
        snapshot = snapshot.model_copy(
            update={"capability_profile": snapshot.capability_profile.model_copy(update=profile_update)}
        )
    return snapshot


def _profile_with_dimensions(dims: list[tuple[CapabilityDimension, DimensionScoreStatus]]) -> CapabilityProfile:
    return CapabilityProfile(
        scoring_policy_id="llmtrace-capability-v1",
        scoring_policy_version="0.1.0",
        dimensions=tuple(
            DimensionScoreResult(
                dimension=dim,
                status=status,
                raw_normalized_score=0.5,
            )
            for dim, status in dims
        ),
        coverage_weight=0.75,
    )


def _build_set(
    builder: ReferenceSetBuilder,
    snapshots: list[ReferenceSnapshot],
    *,
    sha: str = "c" * 64,
    set_id: str = "quick-v1-set",
    set_version: str = "0.1.0",
    created_at: datetime | None = None,
) -> ReferenceSet:
    return builder.build(
        reference_set_id=set_id,
        reference_set_version=set_version,
        created_at=created_at or datetime(2026, 8, 20, tzinfo=UTC),
        snapshots=snapshots,
        snapshot_sha256s={s.snapshot_id: sha for s in snapshots},
    )


@pytest.fixture
def builder() -> ReferenceSetBuilder:
    return ReferenceSetBuilder()


@pytest.fixture
def set_repo(tmp_path: Path) -> ReferenceSetRepository:
    return ReferenceSetRepository(directory=tmp_path / "references" / "sets")


# ---------------------------------------------------------------------------
# Identity validation
# ---------------------------------------------------------------------------


class TestIdentityValidation:
    def test_safe_set_id(self, builder: ReferenceSetBuilder) -> None:
        ref_set = _build_set(builder, [_snapshot()], set_id="quick-v1-set")
        assert ref_set.reference_set_id == "quick-v1-set"

    def test_safe_set_version(self, builder: ReferenceSetBuilder) -> None:
        ref_set = _build_set(builder, [_snapshot()], set_version="0.1.0-rc1")
        assert ref_set.reference_set_version == "0.1.0-rc1"

    @pytest.mark.parametrize(
        "bad_id",
        ["../evil", "a/b", "a\\b", "", ".", "..", "a b", "a;b", "a#b"],
    )
    def test_path_traversal_id_rejected(self, builder: ReferenceSetBuilder, bad_id: str) -> None:
        with pytest.raises(ValueError):
            _build_set(builder, [_snapshot()], set_id=bad_id)

    @pytest.mark.parametrize("bad_version", ["../x", "a/b", "a\\b", "", ".", "a b"])
    def test_path_traversal_version_rejected(self, builder: ReferenceSetBuilder, bad_version: str) -> None:
        with pytest.raises(ValueError):
            _build_set(builder, [_snapshot()], set_version=bad_version)


# ---------------------------------------------------------------------------
# Builder basics
# ---------------------------------------------------------------------------


class TestBuilderBasics:
    def test_empty_set_rejected(self, builder: ReferenceSetBuilder) -> None:
        with pytest.raises(ReferenceSetError):
            _build_set(builder, [])

    def test_duplicate_snapshot_id_rejected(self, builder: ReferenceSetBuilder) -> None:
        with pytest.raises(ReferenceSetError):
            _build_set(builder, [_snapshot("snap-a"), _snapshot("snap-a")])

    def test_missing_sha_rejected(self, builder: ReferenceSetBuilder) -> None:
        with pytest.raises(ReferenceSetError):
            builder.build(
                reference_set_id="quick-v1-set",
                reference_set_version="0.1.0",
                created_at=datetime(2026, 8, 20, tzinfo=UTC),
                snapshots=[_snapshot("snap-a"), _snapshot("snap-b")],
                snapshot_sha256s={"snap-a": "c" * 64},
            )

    def test_naive_created_at_rejected(self, builder: ReferenceSetBuilder) -> None:
        with pytest.raises(ValueError):
            _build_set(builder, [_snapshot()], created_at=datetime(2026, 8, 20))

    def test_members_sorted_deterministically(self, builder: ReferenceSetBuilder) -> None:
        ref_set = _build_set(builder, [_snapshot("snap-b"), _snapshot("snap-a")])
        assert [m.snapshot_id for m in ref_set.members] == ["snap-a", "snap-b"]

    def test_same_model_multiple_snapshots_allowed(self, builder: ReferenceSetBuilder) -> None:
        ref_set = _build_set(builder, [_snapshot("snap-a", model_id="gpt-x"), _snapshot("snap-b", model_id="gpt-x")])
        assert len(ref_set.members) == 2


# ---------------------------------------------------------------------------
# test_fixture handling
# ---------------------------------------------------------------------------


class TestTestFixture:
    def test_test_fixture_rejected_from_trusted_set(self) -> None:
        fixture = _snapshot("fixture-1", provenance_update={"source_type": "test_fixture"})
        with pytest.raises(FixtureReferenceError):
            _build_set(ReferenceSetBuilder(), [fixture])

    def test_test_fixture_allowed_with_flag(self) -> None:
        fixture = _snapshot("fixture-1", provenance_update={"source_type": "test_fixture"})
        ref_set = _build_set(ReferenceSetBuilder(allow_test_fixture=True), [fixture])
        assert ref_set.members[0].snapshot_id == "fixture-1"


# ---------------------------------------------------------------------------
# Content self-checksum
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_content_hash_roundtrip(self, builder: ReferenceSetBuilder) -> None:
        ref_set = _build_set(builder, [_snapshot("snap-a")])
        assert ref_set.compute_content_sha256() == ref_set.content_sha256
        assert ref_set.verify_content_hash() == ref_set.content_sha256

    def test_tampered_set_rejected(self, builder: ReferenceSetBuilder) -> None:
        ref_set = _build_set(builder, [_snapshot("snap-a")])
        tampered = ref_set.model_copy(update={"description": "tampered"})
        with pytest.raises(ReferenceSetIntegrityError):
            tampered.verify_content_hash()

    def test_content_hash_stable_across_reconstruction(self, builder: ReferenceSetBuilder) -> None:
        first = _build_set(builder, [_snapshot("snap-a")])
        second = _build_set(builder, [_snapshot("snap-a")])
        assert first.content_sha256 == second.content_sha256

    def test_content_hash_changes_with_membership(self, builder: ReferenceSetBuilder) -> None:
        one = _build_set(builder, [_snapshot("snap-a")])
        two = _build_set(builder, [_snapshot("snap-a"), _snapshot("snap-b")])
        assert one.content_sha256 != two.content_sha256

    def test_content_hash_format(self, builder: ReferenceSetBuilder) -> None:
        ref_set = _build_set(builder, [_snapshot("snap-a")])
        assert len(ref_set.content_sha256) == 64
        assert ref_set.content_sha256 == ref_set.content_sha256.lower()


# ---------------------------------------------------------------------------
# 12-gate compatibility
# ---------------------------------------------------------------------------


class TestCompatibility:
    def _compat_fail(
        self,
        builder: ReferenceSetBuilder,
        base: ReferenceSnapshot,
        other: ReferenceSnapshot,
    ) -> None:
        with pytest.raises(ReferenceSetCompatibilityError):
            _build_set(builder, [base, other])

    def test_suite_id_mismatch(self, builder: ReferenceSetBuilder) -> None:
        base = _snapshot("snap-a")
        other = _snapshot("snap-b").model_copy(update={"suite_id": "llmtrace_quick_v2"})
        self._compat_fail(builder, base, other)

    def test_suite_version_mismatch(self, builder: ReferenceSetBuilder) -> None:
        base = _snapshot("snap-a")
        other = _snapshot("snap-b").model_copy(update={"suite_version": "0.2.0"})
        self._compat_fail(builder, base, other)

    def test_suite_content_sha256_mismatch(self, builder: ReferenceSetBuilder) -> None:
        base = _snapshot("snap-a")
        other = _snapshot("snap-b", provenance_update={"suite_sha256": "d" * 64})
        self._compat_fail(builder, base, other)

    def test_adapter_id_mismatch(self, builder: ReferenceSetBuilder) -> None:
        base = _snapshot("snap-a")
        other = _snapshot("snap-b", provenance_update={"adapter_id": "llmtrace-quick-v2"})
        self._compat_fail(builder, base, other)

    def test_adapter_version_mismatch(self, builder: ReferenceSetBuilder) -> None:
        base = _snapshot("snap-a")
        other = _snapshot("snap-b", provenance_update={"adapter_version": "0.2.0"})
        self._compat_fail(builder, base, other)

    def test_scoring_policy_id_mismatch(self, builder: ReferenceSetBuilder) -> None:
        base = _snapshot("snap-a")
        other = _snapshot("snap-b", profile_update={"scoring_policy_id": "other-policy"})
        self._compat_fail(builder, base, other)

    def test_scoring_policy_version_mismatch(self, builder: ReferenceSetBuilder) -> None:
        base = _snapshot("snap-a")
        other = _snapshot("snap-b", profile_update={"scoring_policy_version": "0.2.0"})
        self._compat_fail(builder, base, other)

    def test_generation_config_mismatch(self, builder: ReferenceSetBuilder) -> None:
        base = _snapshot("snap-a")
        other = _snapshot("snap-b", provenance_update={"generation_config_sha256": "e" * 64})
        self._compat_fail(builder, base, other)

    def test_qualification_policy_id_mismatch(self, builder: ReferenceSetBuilder) -> None:
        base = _snapshot("snap-a")
        other = _snapshot("snap-b", provenance_update={"qualification_policy_id": "other-qual"})
        self._compat_fail(builder, base, other)

    def test_qualification_policy_version_mismatch(self, builder: ReferenceSetBuilder) -> None:
        base = _snapshot("snap-a")
        other = _snapshot("snap-b", provenance_update={"qualification_policy_version": "0.2.0"})
        self._compat_fail(builder, base, other)

    def test_dimension_set_mismatch(self, builder: ReferenceSetBuilder) -> None:
        base = _snapshot("snap-a")
        other_profile = _profile_with_dimensions(
            [
                (CapabilityDimension.REASONING, DimensionScoreStatus.UNCALIBRATED),
                (CapabilityDimension.CODING, DimensionScoreStatus.UNCALIBRATED),
            ]
        )
        other = _snapshot("snap-b").model_copy(update={"capability_profile": other_profile})
        self._compat_fail(builder, base, other)

    def test_coverage_weight_mismatch(self, builder: ReferenceSetBuilder) -> None:
        base = _snapshot("snap-a")
        other = _snapshot("snap-b", profile_update={"coverage_weight": 0.5})
        self._compat_fail(builder, base, other)

    def test_missing_provenance_field_rejected(self, builder: ReferenceSetBuilder) -> None:
        base = _snapshot("snap-a")
        other = _snapshot("snap-b", provenance_update={"adapter_id": None})
        with pytest.raises(ReferenceSetCompatibilityError):
            _build_set(builder, [base, other])

    def test_compatible_snapshots_build(self, builder: ReferenceSetBuilder) -> None:
        ref_set = _build_set(builder, [_snapshot("snap-a"), _snapshot("snap-b")])
        assert len(ref_set.members) == 2
        assert ref_set.suite_id == "llmtrace_quick_v1"
        assert ref_set.suite_content_sha256 == _snapshot("snap-a").provenance.suite_sha256


# ---------------------------------------------------------------------------
# Repository — append-only
# ---------------------------------------------------------------------------


class TestRepository:
    def test_save_and_get_roundtrip(self, set_repo: ReferenceSetRepository, builder: ReferenceSetBuilder) -> None:
        ref_set = _build_set(builder, [_snapshot("snap-a")])
        saved = set_repo.save(ref_set)
        assert set_repo.get("quick-v1-set", "0.1.0") == saved

    def test_duplicate_save_rejected(self, set_repo: ReferenceSetRepository, builder: ReferenceSetBuilder) -> None:
        ref_set = _build_set(builder, [_snapshot("snap-a")])
        set_repo.save(ref_set)
        with pytest.raises(DuplicateReferenceSetError):
            set_repo.save(ref_set)

    def test_new_version_is_new_record(self, set_repo: ReferenceSetRepository, builder: ReferenceSetBuilder) -> None:
        set_repo.save(_build_set(builder, [_snapshot("snap-a")], set_version="0.1.0"))
        set_repo.save(_build_set(builder, [_snapshot("snap-a")], set_version="0.2.0"))
        assert len(set_repo) == 2

    def test_get_not_found(self, set_repo: ReferenceSetRepository) -> None:
        with pytest.raises(ReferenceSetNotFoundError):
            set_repo.get("nope", "0.1.0")

    def test_list_sorted(self, set_repo: ReferenceSetRepository, builder: ReferenceSetBuilder) -> None:
        set_repo.save(_build_set(builder, [_snapshot("snap-a")], set_id="b-set", set_version="0.1.0"))
        set_repo.save(_build_set(builder, [_snapshot("snap-a")], set_id="a-set", set_version="0.2.0"))
        listed = set_repo.list()
        assert [(s.reference_set_id, s.reference_set_version) for s in listed] == [
            ("a-set", "0.2.0"),
            ("b-set", "0.1.0"),
        ]

    def test_load_from_disk_and_verify(self, tmp_path: Path, builder: ReferenceSetBuilder) -> None:
        repo = ReferenceSetRepository(directory=tmp_path / "sets")
        ref_set = _build_set(builder, [_snapshot("snap-a")])
        repo.save(ref_set)
        reloaded = ReferenceSetRepository.load(tmp_path / "sets")
        loaded = reloaded.get("quick-v1-set", "0.1.0")
        assert loaded == ref_set
        assert reloaded.verify("quick-v1-set", "0.1.0") == repo.set_sha256("quick-v1-set", "0.1.0")

    def test_tampered_set_file_rejected(self, tmp_path: Path, builder: ReferenceSetBuilder) -> None:
        repo = ReferenceSetRepository(directory=tmp_path / "sets")
        ref_set = _build_set(builder, [_snapshot("snap-a")])
        repo.save(ref_set)
        # Tamper the on-disk file in a JSON-valid way: content no longer
        # matches the declared content_sha256.
        path = tmp_path / "sets" / "quick-v1-set_0.1.0.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["description"] = "tampered"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        reloaded = ReferenceSetRepository.load(tmp_path / "sets")
        loaded = reloaded.get("quick-v1-set", "0.1.0")
        with pytest.raises(ReferenceSetIntegrityError):
            loaded.verify_content_hash()
        with pytest.raises(ReferenceSetIntegrityError):
            reloaded.verify("quick-v1-set", "0.1.0")
