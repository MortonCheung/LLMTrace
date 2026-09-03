"""Adversarial + happy-path tests for the shared ReferenceSet calibration
validator (``reference/validation.py``, v0.4-B).

These cover the owner-confirmed correctness blockers B3 / B4 at the
validation layer and the CLI/runner shared path:

    D. forged ReferenceSet self-hash with wrong member snapshot SHA
    E. member capability_profile SHA mismatch (run artifact intact)
    F. missing trusted sidecar (legacy v0.3-C snapshot)
    G. test_fixture source_type refused
    H. suite content SHA mismatch
    I. generation config SHA mismatch
    J. scoring policy mismatch
    K. real trust chain happy path

Every set under test is produced by the production builders via
``build_trusted_reference_root`` — no ``model_construct``, no fake SHA-256s.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmtrace.execution.artifacts import RunArtifactRepository
from llmtrace.reference.reference_set import ReferenceSet
from llmtrace.reference.validation import (
    CalibrationContext,
    validate_reference_set_for_calibration,
)
from llmtrace.scoring.calibration import (
    ReferenceSetIncompatibleError,
    ReferenceSetIntegrityFailureError,
    UntrustedReferenceSourceError,
)
from llmtrace.scoring.reference import ReferenceRepository

from .helpers import build_trusted_reference_root


def _artifact_repository(root: Path) -> RunArtifactRepository:
    """``build_trusted_reference_root`` commits at ``root/runs`` — the same
    repository the CLI / runner opens from ``--output-dir root``."""
    return RunArtifactRepository(root)


def _rehash_set(set_path: Path) -> ReferenceSet:
    """Reload a set file, optionally modified, and re-stamp its self-hash.

    Re-stamping is exactly what a forger would do after editing the JSON:
    ``verify_content_hash`` alone cannot catch a *consistent* tamper — only
    the trust-chain re-verification can.
    """
    reference_set = ReferenceSet.model_validate_json(set_path.read_text(encoding="utf-8"))
    return reference_set.model_copy(update={"content_sha256": reference_set.compute_content_sha256()})


def _write_rehashed(set_path: Path, mutate) -> Path:
    reference_set = _rehash_set(set_path)
    mutated = mutate(reference_set)
    forged = mutated.model_copy(update={"content_sha256": mutated.compute_content_sha256()})
    forged_path = set_path.with_name("forged_set.json")
    forged_path.write_text(forged.model_dump_json(indent=2))
    return forged_path


class TestReferenceSetValidation:
    def test_happy_path_returns_verified_context(self, tmp_path: Path) -> None:
        """K: the full real trust chain must validate and yield a context."""
        ref_root, set_path = build_trusted_reference_root(tmp_path)
        context = validate_reference_set_for_calibration(
            set_path=set_path,
            artifact_repository=_artifact_repository(tmp_path),
        )
        assert isinstance(context, CalibrationContext)
        assert context.reference_set.reference_set_id == "calib-set"
        assert context.reference_set.content_sha256 == context.reference_set.compute_content_sha256()
        assert len(context.reference_set.members) == 5
        assert context.calibration_policy.policy_id == "llmtrace-reference-calibration-v1"
        # Snapshots were re-verified from disk inside the validator.
        snap_repo = ReferenceRepository.load(ref_root / "snapshots")
        assert len(snap_repo) == 5

    def test_set_outside_repository_layout_fails_closed(self, tmp_path: Path) -> None:
        """§5: a set path that cannot map onto <root>/sets/... is refused."""
        ref_root, set_path = build_trusted_reference_root(tmp_path)
        relocated = tmp_path / "elsewhere" / "set.json"
        relocated.parent.mkdir()
        relocated.write_text(set_path.read_text(encoding="utf-8"))
        with pytest.raises(ReferenceSetIntegrityFailureError):
            validate_reference_set_for_calibration(
                set_path=relocated,
                artifact_repository=_artifact_repository(tmp_path),
            )

    def test_forged_self_hash_with_wrong_member_snapshot_sha(self, tmp_path: Path) -> None:
        """D: valid JSON + recomputed content hash, but member.snapshot_sha256
        points at different bytes — self-hash is NOT trust."""
        _, set_path = build_trusted_reference_root(tmp_path)

        def _mutate(reference_set: ReferenceSet) -> ReferenceSet:
            members = list(reference_set.members)
            members[0] = members[0].model_copy(update={"snapshot_sha256": "1" * 64})
            return reference_set.model_copy(update={"members": tuple(members)})

        forged_path = _write_rehashed(set_path, _mutate)
        with pytest.raises(ReferenceSetIntegrityFailureError, match="snapshot SHA-256 binding"):
            validate_reference_set_for_calibration(
                set_path=forged_path,
                artifact_repository=_artifact_repository(tmp_path),
            )

    def test_member_capability_profile_sha_mismatch(self, tmp_path: Path) -> None:
        """E: the run artifact is intact, but the member points at another
        profile SHA — the provenance chain must break."""
        _, set_path = build_trusted_reference_root(tmp_path)

        def _mutate(reference_set: ReferenceSet) -> ReferenceSet:
            members = list(reference_set.members)
            members[0] = members[0].model_copy(update={"capability_profile_sha256": "2" * 64})
            return reference_set.model_copy(update={"members": tuple(members)})

        forged_path = _write_rehashed(set_path, _mutate)
        with pytest.raises(ReferenceSetIntegrityFailureError, match="capability_profile SHA-256 chain"):
            validate_reference_set_for_calibration(
                set_path=forged_path,
                artifact_repository=_artifact_repository(tmp_path),
            )

    def test_missing_trusted_sidecar_fails_closed(self, tmp_path: Path) -> None:
        """F: legacy v0.3-C snapshot — body exists, manifest sidecar missing."""
        ref_root, set_path = build_trusted_reference_root(tmp_path)
        (ref_root / "snapshots" / "snap-0.manifest.json").unlink()
        with pytest.raises(
            ReferenceSetIntegrityFailureError, match="no integrity manifest|trusted snapshot verification"
        ):
            validate_reference_set_for_calibration(
                set_path=set_path,
                artifact_repository=_artifact_repository(tmp_path),
            )

    def test_test_fixture_source_type_refused(self, tmp_path: Path) -> None:
        """G: a TEST_FIXTURE snapshot must fail the trusted source gate."""
        _, set_path = build_trusted_reference_root(
            tmp_path,
            source_type="test_fixture",  # builder allows fixtures; the validator must not
        )
        with pytest.raises(UntrustedReferenceSourceError):
            validate_reference_set_for_calibration(
                set_path=set_path,
                artifact_repository=_artifact_repository(tmp_path),
            )

    def test_suite_content_sha_mismatch(self, tmp_path: Path) -> None:
        """H: suite_content_sha256 differs — even with a legit re-stamped
        self-hash, compatibility fails."""
        _, set_path = build_trusted_reference_root(tmp_path)
        forged_path = _write_rehashed(
            set_path,
            lambda s: s.model_copy(update={"suite_content_sha256": "3" * 64}),
        )
        with pytest.raises(ReferenceSetIncompatibleError, match="suite_content_sha256"):
            validate_reference_set_for_calibration(
                set_path=forged_path,
                artifact_repository=_artifact_repository(tmp_path),
            )

    def test_generation_config_sha_mismatch(self, tmp_path: Path) -> None:
        """I: generation_config_sha256 differs — fails closed."""
        _, set_path = build_trusted_reference_root(tmp_path)
        forged_path = _write_rehashed(
            set_path,
            lambda s: s.model_copy(update={"generation_config_sha256": "4" * 64}),
        )
        with pytest.raises(ReferenceSetIncompatibleError, match="generation_config_sha256"):
            validate_reference_set_for_calibration(
                set_path=forged_path,
                artifact_repository=_artifact_repository(tmp_path),
            )

    def test_scoring_policy_mismatch(self, tmp_path: Path) -> None:
        """J: scoring policy differs — fails closed."""
        _, set_path = build_trusted_reference_root(tmp_path)
        forged_path = _write_rehashed(
            set_path,
            lambda s: s.model_copy(update={"scoring_policy_id": "not-the-policy"}),
        )
        with pytest.raises(ReferenceSetIncompatibleError, match="scoring_policy_id"):
            validate_reference_set_for_calibration(
                set_path=forged_path,
                artifact_repository=_artifact_repository(tmp_path),
            )

    def test_adapter_mismatch(self, tmp_path: Path) -> None:
        """§9.4: adapter identity differs — fails closed."""
        _, set_path = build_trusted_reference_root(tmp_path)
        forged_path = _write_rehashed(
            set_path,
            lambda s: s.model_copy(update={"adapter_id": "someone-elses-adapter"}),
        )
        with pytest.raises(ReferenceSetIncompatibleError, match="adapter_id"):
            validate_reference_set_for_calibration(
                set_path=forged_path,
                artifact_repository=_artifact_repository(tmp_path),
            )

    def test_malformed_json_fails_closed(self, tmp_path: Path) -> None:
        """A ReferenceSet that is not even valid JSON is an integrity failure."""
        _, set_path = build_trusted_reference_root(tmp_path)
        bad = tmp_path / "references" / "sets" / "bad.json"
        bad.write_text("{not json")
        with pytest.raises(ReferenceSetIntegrityFailureError):
            validate_reference_set_for_calibration(
                set_path=bad,
                artifact_repository=_artifact_repository(tmp_path),
            )

    def test_self_hash_mismatch_fails_closed(self, tmp_path: Path) -> None:
        """A tampered set whose self-hash no longer recomputes is rejected."""
        _, set_path = build_trusted_reference_root(tmp_path)
        data = json.loads(set_path.read_text(encoding="utf-8"))
        data["description"] = "tampered without re-hashing"
        bad = tmp_path / "references" / "sets" / "bad.json"
        bad.write_text(json.dumps(data, indent=2))
        with pytest.raises(ReferenceSetIntegrityFailureError):
            validate_reference_set_for_calibration(
                set_path=bad,
                artifact_repository=_artifact_repository(tmp_path),
            )
