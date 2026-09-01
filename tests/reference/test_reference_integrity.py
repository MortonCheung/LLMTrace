"""Adversarial regression tests for v0.4-A trusted reference integrity.

Covers Blocker 1 (sidecar integrity anchor), Blocker 2 (URL console scrub),
Blocker 3 (Gate 10 exact dimension set), and Hardening 1 (malformed manifest
fail-closed).  Each test simulates a specific attack or corruption scenario
and asserts fail-closed behavior.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from llmtrace.reference.qualification import (
    ARTIFACT_INTEGRITY_FAILURE,
    INCOMPATIBLE_COVERAGE,
    qualify_reference_run,
)
from llmtrace.scoring.errors import (
    ReferenceIntegrityError,
    ReferenceSnapshotManifestMissingError,
    ReferenceSnapshotProvenanceMismatchError,
)
from llmtrace.scoring.models import (
    CapabilityDimension,
    CapabilityProfile,
    DimensionScoreResult,
    DimensionScoreStatus,
)
from llmtrace.scoring.reference import ReferenceRepository, ReferenceSnapshot
from llmtrace.security.redaction import redact_url

from .helpers import DEFAULT_EXECUTION_ID, commit_run, make_capability_profile, make_manifest, make_snapshot

# ===========================================================================
# Blocker 1 — Sidecar integrity anchor
# ===========================================================================


class TestSidecarIntegrityAnchor:
    """Case A-D: pre-load tamper / post-save tamper / sidecar tamper / legacy."""

    def test_case_a_pre_load_tamper_rejected_by_set_create(self, tmp_path: Path) -> None:
        """Case A: tamper snapshot.json before load → set-create must fail."""
        repo_dir = tmp_path / "snapshots"
        repo = ReferenceRepository(directory=repo_dir)
        snapshot = make_snapshot(snapshot_id="trusted-snap")
        repo.save_trusted(snapshot)

        # Tamper the snapshot body before any fresh load.
        snapshot_path = repo_dir / "trusted-snap.json"
        tampered = json.loads(snapshot_path.read_text(encoding="utf-8"))
        tampered["model_id"] = "tampered-model"
        snapshot_path.write_text(json.dumps(tampered, indent=2), encoding="utf-8")

        # Fresh load reads tampered content.
        fresh_repo = ReferenceRepository.load(repo_dir)
        loaded = fresh_repo.get("trusted-snap")
        assert loaded.model_id == "tampered-model"

        # verify_trusted_snapshot must detect the mismatch.
        with pytest.raises(ReferenceIntegrityError, match="integrity check failed"):
            fresh_repo.verify_trusted_snapshot("trusted-snap")

    def test_case_b_post_save_body_tamper_rejected(self, tmp_path: Path) -> None:
        """Case B: modify snapshot.json, leave sidecar intact → verify fails."""
        repo = ReferenceRepository(directory=tmp_path)
        snapshot = make_snapshot(snapshot_id="snap-b")
        repo.save_trusted(snapshot)

        # Tamper body after save.
        snapshot_path = tmp_path / "snap-b.json"
        tampered = json.loads(snapshot_path.read_text(encoding="utf-8"))
        tampered["provider_id"] = "evil-provider"
        snapshot_path.write_text(json.dumps(tampered, indent=2), encoding="utf-8")

        with pytest.raises(ReferenceIntegrityError, match="integrity check failed"):
            repo.verify_snapshot("snap-b")

    def test_case_c_sidecar_sha_tamper_rejected(self, tmp_path: Path) -> None:
        """Case C: modify sidecar.snapshot_sha256 → verify fails."""
        repo = ReferenceRepository(directory=tmp_path)
        snapshot = make_snapshot(snapshot_id="snap-c")
        repo.save_trusted(snapshot)

        sidecar_path = tmp_path / "snap-c.manifest.json"
        sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar_data["snapshot_sha256"] = "f" * 64
        sidecar_path.write_text(json.dumps(sidecar_data, indent=2), encoding="utf-8")

        with pytest.raises(ReferenceIntegrityError, match="integrity check failed"):
            repo.verify_snapshot("snap-c")

    def test_case_d_legacy_snapshot_readable_but_set_rejected(self, tmp_path: Path) -> None:
        """Case D: v0.3-C legacy (no sidecar) → readable, but set-create rejects."""
        repo = ReferenceRepository(directory=tmp_path)
        snapshot = make_snapshot(snapshot_id="legacy-snap")
        repo.save(snapshot)  # Old save path — no sidecar.

        # Legacy snapshot remains readable.
        loaded = repo.get("legacy-snap")
        assert loaded.snapshot_id == "legacy-snap"

        # verify_snapshot with no expected falls back to current serialization.
        actual_sha = repo.verify_snapshot("legacy-snap")
        assert len(actual_sha) == 64

        # verify_trusted_snapshot demands a sidecar.
        with pytest.raises(ReferenceSnapshotManifestMissingError, match="no integrity manifest"):
            repo.verify_trusted_snapshot("legacy-snap")

    def test_sidecar_provenance_mismatch_rejected(self, tmp_path: Path) -> None:
        """Tamper snapshot provenance → sidecar binding check fails."""
        repo = ReferenceRepository(directory=tmp_path)
        snapshot = make_snapshot(snapshot_id="snap-prov")
        repo.save_trusted(snapshot)

        # Tamper provenance hash in snapshot body.
        snapshot_path = tmp_path / "snap-prov.json"
        tampered = json.loads(snapshot_path.read_text(encoding="utf-8"))
        tampered["provenance"]["run_manifest_sha256"] = "e" * 64
        snapshot_path.write_text(json.dumps(tampered, indent=2), encoding="utf-8")

        # Fresh load picks up tampered provenance.
        fresh_repo = ReferenceRepository.load(tmp_path)

        with pytest.raises(ReferenceSnapshotProvenanceMismatchError, match="provenance disagrees"):
            fresh_repo.verify_snapshot("snap-prov")

    def test_sidecar_missing_provenance_fields_rejected_at_save(self, tmp_path: Path) -> None:
        """A snapshot missing provenance hashes cannot be saved as trusted."""
        from llmtrace.scoring.reference import ReferenceProvenance

        incomplete_provenance = ReferenceProvenance(
            source_type="operator_verified_api_run",
            created_by="operator",
            created_at=datetime(2026, 8, 15, tzinfo=UTC),
            suite_sha256="a" * 64,
            benchmark_revision="quick-v1-rev",
            runner_version="0.4.0",
            execution_id=None,  # Missing
            run_manifest_sha256=None,  # Missing
            capability_profile_sha256=None,  # Missing
        )
        snapshot = ReferenceSnapshot(
            snapshot_id="incomplete-snap",
            model_id="test-model",
            provider_id="test-provider",
            created_at=datetime(2026, 8, 15, tzinfo=UTC),
            suite_id="llmtrace_quick_v1",
            suite_version="0.1.0",
            capability_profile=make_capability_profile(),
            provenance=incomplete_provenance,
        )
        repo = ReferenceRepository(directory=tmp_path)
        with pytest.raises(ReferenceSnapshotProvenanceMismatchError, match="provenance field"):
            repo.save_trusted(snapshot)


# ===========================================================================
# Blocker 2 — URL console scrub
# ===========================================================================


class TestURLConsoleScrub:
    """Verify redact_url scrubs userinfo, secret query params, preserves region."""

    def test_userinfo_credentials_redacted(self) -> None:
        url = "https://myuser:mypassword@example.com/v1"
        redacted = redact_url(url)
        assert "myuser" not in redacted
        assert "mypassword" not in redacted
        assert "[REDACTED]@example.com" in redacted

    def test_sensitive_query_token_redacted(self) -> None:
        url = "https://example.com/v1?token=very-secret"
        redacted = redact_url(url)
        assert "very-secret" not in redacted
        assert "token=[REDACTED]" in redacted

    def test_sensitive_query_api_key_redacted(self) -> None:
        url = "https://example.com/v1?api_key=secret-key-123"
        redacted = redact_url(url)
        assert "secret-key-123" not in redacted
        assert "api_key=[REDACTED]" in redacted

    def test_non_sensitive_query_region_preserved(self) -> None:
        url = "https://example.com/v1?region=us-west&zone=a"
        redacted = redact_url(url)
        assert "region=us-west" in redacted
        assert "zone=a" in redacted

    def test_mixed_credentials_and_non_sensitive_query(self) -> None:
        url = "https://admin:secret@example.com/v1?api_key=key123&region=us"
        redacted = redact_url(url)
        assert "admin" not in redacted
        assert "secret" not in redacted
        assert "key123" not in redacted
        assert "region=us" in redacted
        assert "[REDACTED]@example.com" in redacted


# ===========================================================================
# Blocker 3 — Gate 10 exact dimension set
# ===========================================================================


class TestGate10ExactDimensionSet:
    """Extra SCORED/UNCALIBRATED dimension must be rejected."""

    def test_extra_scored_dimension_rejected(self, artifact_root: Path) -> None:
        # Policy enables 4 dimensions; profile has 5 (4 expected + 1 extra SCORED).
        profile = CapabilityProfile(
            scoring_policy_id="llmtrace-capability-v1",
            scoring_policy_version="0.1.0",
            dimensions=(
                DimensionScoreResult(
                    dimension=CapabilityDimension.REASONING,
                    status=DimensionScoreStatus.UNCALIBRATED,
                    raw_normalized_score=0.5,
                ),
                DimensionScoreResult(
                    dimension=CapabilityDimension.CODING,
                    status=DimensionScoreStatus.UNCALIBRATED,
                    raw_normalized_score=0.5,
                ),
                DimensionScoreResult(
                    dimension=CapabilityDimension.MATH_SCIENCE,
                    status=DimensionScoreStatus.UNCALIBRATED,
                    raw_normalized_score=0.5,
                ),
                DimensionScoreResult(
                    dimension=CapabilityDimension.INSTRUCTION_FOLLOWING,
                    status=DimensionScoreStatus.UNCALIBRATED,
                    raw_normalized_score=0.5,
                ),
                DimensionScoreResult(
                    dimension=CapabilityDimension.DATA_ANALYSIS,
                    status=DimensionScoreStatus.SCORED,
                    raw_normalized_score=0.75,
                ),
            ),
            coverage_weight=0.75,
        )
        repository, _ = commit_run(artifact_root, profile=profile)
        result = qualify_reference_run(
            execution_id=DEFAULT_EXECUTION_ID,
            artifact_repository=repository,
        )
        assert not result.qualified
        assert INCOMPATIBLE_COVERAGE in result.reason_codes

    def test_extra_uncalibrated_dimension_rejected(self, artifact_root: Path) -> None:
        # 5 UNCALIBRATED dimensions when policy only enables 4.
        profile = CapabilityProfile(
            scoring_policy_id="llmtrace-capability-v1",
            scoring_policy_version="0.1.0",
            dimensions=(
                DimensionScoreResult(
                    dimension=CapabilityDimension.REASONING,
                    status=DimensionScoreStatus.UNCALIBRATED,
                    raw_normalized_score=0.5,
                ),
                DimensionScoreResult(
                    dimension=CapabilityDimension.CODING,
                    status=DimensionScoreStatus.UNCALIBRATED,
                    raw_normalized_score=0.5,
                ),
                DimensionScoreResult(
                    dimension=CapabilityDimension.MATH_SCIENCE,
                    status=DimensionScoreStatus.UNCALIBRATED,
                    raw_normalized_score=0.5,
                ),
                DimensionScoreResult(
                    dimension=CapabilityDimension.INSTRUCTION_FOLLOWING,
                    status=DimensionScoreStatus.UNCALIBRATED,
                    raw_normalized_score=0.5,
                ),
                DimensionScoreResult(
                    dimension=CapabilityDimension.DATA_ANALYSIS,
                    status=DimensionScoreStatus.UNCALIBRATED,
                    raw_normalized_score=0.6,
                ),
            ),
            coverage_weight=0.75,
        )
        repository, _ = commit_run(artifact_root, profile=profile)
        result = qualify_reference_run(
            execution_id=DEFAULT_EXECUTION_ID,
            artifact_repository=repository,
        )
        assert not result.qualified
        assert INCOMPATIBLE_COVERAGE in result.reason_codes


# ===========================================================================
# Hardening 1 — Malformed manifest fail-closed
# ===========================================================================


class TestMalformedManifestFailClosed:
    """Invalid JSON / missing field / schema violation → REJECTED + ARTIFACT_INTEGRITY_FAILURE."""

    def test_invalid_json_manifest_rejected(self, artifact_root: Path) -> None:
        from llmtrace.execution.artifacts import RunArtifactRepository

        repository = RunArtifactRepository(artifact_root)
        manifest = make_manifest()
        repository.commit(manifest, {"capability_profile.json": make_capability_profile().model_dump_json()})

        # Corrupt manifest to invalid JSON.
        manifest_path = artifact_root / "runs" / DEFAULT_EXECUTION_ID / "manifest.json"
        manifest_path.write_text("{invalid json", encoding="utf-8")

        result = qualify_reference_run(
            execution_id=DEFAULT_EXECUTION_ID,
            artifact_repository=repository,
        )
        assert not result.qualified
        assert ARTIFACT_INTEGRITY_FAILURE in result.reason_codes
        assert any("unreadable or malformed" in w for w in result.warnings)

    def test_missing_required_field_rejected(self, artifact_root: Path) -> None:
        from llmtrace.execution.artifacts import RunArtifactRepository

        repository = RunArtifactRepository(artifact_root)
        manifest = make_manifest()
        repository.commit(manifest, {"capability_profile.json": make_capability_profile().model_dump_json()})

        # Remove a required field from manifest.
        manifest_path = artifact_root / "runs" / DEFAULT_EXECUTION_ID / "manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        del data["execution_id"]
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        result = qualify_reference_run(
            execution_id=DEFAULT_EXECUTION_ID,
            artifact_repository=repository,
        )
        assert not result.qualified
        assert ARTIFACT_INTEGRITY_FAILURE in result.reason_codes

    def test_invalid_adapter_field_rejected(self, artifact_root: Path) -> None:
        from llmtrace.execution.artifacts import RunArtifactRepository

        repository = RunArtifactRepository(artifact_root)
        manifest = make_manifest()
        repository.commit(manifest, {"capability_profile.json": make_capability_profile().model_dump_json()})

        # Set adapter_id to empty string (violates min_length=1).
        manifest_path = artifact_root / "runs" / DEFAULT_EXECUTION_ID / "manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["adapter_id"] = ""
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        result = qualify_reference_run(
            execution_id=DEFAULT_EXECUTION_ID,
            artifact_repository=repository,
        )
        assert not result.qualified
        assert ARTIFACT_INTEGRITY_FAILURE in result.reason_codes
