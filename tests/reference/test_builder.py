"""ReferenceSnapshotBuilder tests (§40): persisted-profile source, provenance completeness."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmtrace.adapters.quick_suite import (
    QUICK_SUITE_SUITE_ID,
    QUICK_SUITE_SUITE_VERSION,
    get_quick_suite_source_revisions,
)
from llmtrace.reference.builder import OPERATOR_VERIFIED_API_RUN, ReferenceSnapshotBuilder
from llmtrace.reference.qualification import ReferenceQualificationError
from llmtrace.scoring.errors import DuplicateSnapshotError, ReferenceIntegrityError
from llmtrace.scoring.reference import ReferenceRepository

from .helpers import DEFAULT_EXECUTION_ID, commit_run


@pytest.fixture
def builder() -> ReferenceSnapshotBuilder:
    return ReferenceSnapshotBuilder()


@pytest.fixture
def artifact_root(tmp_path: Path) -> Path:
    return tmp_path / "reference-runs"


def _run_file(root: Path, execution_id: str, filename: str) -> Path:
    return root / "runs" / execution_id / filename


class TestBuildHappyPath:
    def test_snapshot_saved_with_full_provenance(
        self,
        artifact_root: Path,
        snapshot_repository: ReferenceRepository,
        builder: ReferenceSnapshotBuilder,
    ) -> None:
        repository, _manifest = commit_run(artifact_root, profile_score=0.25)
        snapshot = builder.build(
            execution_id=DEFAULT_EXECUTION_ID,
            artifact_repository=repository,
            reference_repository=snapshot_repository,
            provider_id="openai",
            snapshot_id="openai-ref-v1",
            created_by="operator",
        )

        assert snapshot.snapshot_id == "openai-ref-v1"
        assert snapshot.model_id == "my-real-model"
        assert snapshot.provider_id == "openai"
        assert snapshot.suite_id == QUICK_SUITE_SUITE_ID
        assert snapshot.suite_version == QUICK_SUITE_SUITE_VERSION
        assert snapshot in snapshot_repository.list()

    def test_provenance_records_manifest_facts(
        self,
        artifact_root: Path,
        snapshot_repository: ReferenceRepository,
        builder: ReferenceSnapshotBuilder,
    ) -> None:
        repository, manifest = commit_run(artifact_root)
        snapshot = builder.build(
            execution_id=DEFAULT_EXECUTION_ID,
            artifact_repository=repository,
            reference_repository=snapshot_repository,
            provider_id="openai",
            snapshot_id="openai-ref-v1",
            created_by="operator",
        )

        prov = snapshot.provenance
        assert prov.source_type == OPERATOR_VERIFIED_API_RUN
        assert prov.created_by == "operator"
        assert prov.execution_id == DEFAULT_EXECUTION_ID
        assert prov.endpoint_redacted == manifest.base_url_redacted
        assert prov.adapter_id == manifest.adapter_id
        assert prov.adapter_version == manifest.adapter_version
        assert prov.generation_config_sha256 == manifest.generation_config_sha256
        assert prov.suite_sha256 == manifest.suite_content_sha256
        assert prov.qualification_policy_id == "llmtrace_reference_qualification_v1"
        assert prov.qualification_policy_version == "0.1.0"

    def test_run_manifest_sha256_is_actual_file_bytes(
        self,
        artifact_root: Path,
        snapshot_repository: ReferenceRepository,
        builder: ReferenceSnapshotBuilder,
    ) -> None:
        repository, _ = commit_run(artifact_root)
        snapshot = builder.build(
            execution_id=DEFAULT_EXECUTION_ID,
            artifact_repository=repository,
            reference_repository=snapshot_repository,
            provider_id="openai",
            snapshot_id="openai-ref-v1",
            created_by="operator",
        )
        assert snapshot.provenance.run_manifest_sha256 == repository.manifest_sha256(DEFAULT_EXECUTION_ID)
        assert len(snapshot.provenance.run_manifest_sha256) == 64

    def test_profile_sha256_matches_persisted_artifact(
        self,
        artifact_root: Path,
        snapshot_repository: ReferenceRepository,
        builder: ReferenceSnapshotBuilder,
    ) -> None:
        repository, manifest = commit_run(artifact_root, profile_score=0.25)
        snapshot = builder.build(
            execution_id=DEFAULT_EXECUTION_ID,
            artifact_repository=repository,
            reference_repository=snapshot_repository,
            provider_id="openai",
            snapshot_id="openai-ref-v1",
            created_by="operator",
        )
        recorded = manifest.artifacts["capability_profile.json"]
        assert snapshot.provenance.capability_profile_sha256 == recorded

    def test_profile_comes_from_persisted_artifact(
        self,
        artifact_root: Path,
        snapshot_repository: ReferenceRepository,
        builder: ReferenceSnapshotBuilder,
    ) -> None:
        # score 0.25 flows through the persisted artifact — proving Gate 3.
        repository, _ = commit_run(artifact_root, profile_score=0.25)
        snapshot = builder.build(
            execution_id=DEFAULT_EXECUTION_ID,
            artifact_repository=repository,
            reference_repository=snapshot_repository,
            provider_id="openai",
            snapshot_id="openai-ref-v1",
            created_by="operator",
        )
        assert snapshot.capability_profile.coverage_weight == 0.75
        assert all(d.raw_normalized_score == 0.25 for d in snapshot.capability_profile.dimensions)

    def test_benchmark_revisions_from_suite_manifest(
        self,
        artifact_root: Path,
        snapshot_repository: ReferenceRepository,
        builder: ReferenceSnapshotBuilder,
    ) -> None:
        repository, _ = commit_run(artifact_root)
        snapshot = builder.build(
            execution_id=DEFAULT_EXECUTION_ID,
            artifact_repository=repository,
            reference_repository=snapshot_repository,
            provider_id="openai",
            snapshot_id="openai-ref-v1",
            created_by="operator",
        )
        assert snapshot.provenance.benchmark_revisions == get_quick_suite_source_revisions()
        assert snapshot.provenance.benchmark_revisions["arc_challenge_quick_v1"] == "ARC-Challenge-2018"
        assert snapshot.provenance.benchmark_revisions["humaneval_quick_v1"] == "human-eval-v1-2021"


# ---------------------------------------------------------------------------
# Adversarial
# ---------------------------------------------------------------------------


class TestBuildAdversarial:
    def test_tampered_profile_rejected_no_snapshot(
        self,
        artifact_root: Path,
        snapshot_repository: ReferenceRepository,
        builder: ReferenceSnapshotBuilder,
    ) -> None:
        repository, _ = commit_run(artifact_root)
        path = _run_file(artifact_root, DEFAULT_EXECUTION_ID, "capability_profile.json")
        path.write_text(path.read_text(encoding="utf-8") + "\n# tampered", encoding="utf-8")

        with pytest.raises(ReferenceQualificationError) as exc_info:
            builder.build(
                execution_id=DEFAULT_EXECUTION_ID,
                artifact_repository=repository,
                reference_repository=snapshot_repository,
                provider_id="openai",
                snapshot_id="openai-ref-v1",
                created_by="operator",
            )
        assert exc_info.value.error_code == "ARTIFACT_INTEGRITY_FAILURE"
        assert len(snapshot_repository) == 0

    def test_tampered_manifest_rejected(
        self,
        artifact_root: Path,
        snapshot_repository: ReferenceRepository,
        builder: ReferenceSnapshotBuilder,
    ) -> None:
        repository, _ = commit_run(artifact_root)
        path = _run_file(artifact_root, DEFAULT_EXECUTION_ID, "manifest.json")
        tampered = path.read_text(encoding="utf-8").replace(
            '"suite_id": "llmtrace_quick_v1"', '"suite_id": "llmtrace_quick_v2"'
        )
        path.write_text(tampered, encoding="utf-8")

        with pytest.raises(ReferenceQualificationError) as exc_info:
            builder.build(
                execution_id=DEFAULT_EXECUTION_ID,
                artifact_repository=repository,
                reference_repository=snapshot_repository,
                provider_id="openai",
                snapshot_id="openai-ref-v1",
                created_by="operator",
            )
        assert exc_info.value.error_code == "SUITE_MISMATCH"
        assert len(snapshot_repository) == 0

    def test_duplicate_snapshot_id_rejected(
        self,
        artifact_root: Path,
        snapshot_repository: ReferenceRepository,
        builder: ReferenceSnapshotBuilder,
    ) -> None:
        repository, _ = commit_run(artifact_root)
        builder.build(
            execution_id=DEFAULT_EXECUTION_ID,
            artifact_repository=repository,
            reference_repository=snapshot_repository,
            provider_id="openai",
            snapshot_id="openai-ref-v1",
            created_by="operator",
        )
        with pytest.raises(DuplicateSnapshotError):
            builder.build(
                execution_id=DEFAULT_EXECUTION_ID,
                artifact_repository=repository,
                reference_repository=snapshot_repository,
                provider_id="openai",
                snapshot_id="openai-ref-v1",
                created_by="operator",
            )

    def test_snapshot_persisted_then_tampered_detectable(
        self,
        artifact_root: Path,
        snapshot_repository: ReferenceRepository,
        builder: ReferenceSnapshotBuilder,
    ) -> None:
        repository, _ = commit_run(artifact_root)
        builder.build(
            execution_id=DEFAULT_EXECUTION_ID,
            artifact_repository=repository,
            reference_repository=snapshot_repository,
            provider_id="openai",
            snapshot_id="openai-ref-v1",
            created_by="operator",
        )
        file_path = snapshot_repository._directory / "openai-ref-v1.json"  # type: ignore[union-attr]
        assert file_path is not None and file_path.exists()
        file_path.write_text(file_path.read_text(encoding="utf-8") + "\n# tampered", encoding="utf-8")
        with pytest.raises(ReferenceIntegrityError):
            snapshot_repository.verify_snapshot("openai-ref-v1")


class TestBuilderQualify:
    def test_qualify_returns_qualified_result(
        self,
        artifact_root: Path,
        builder: ReferenceSnapshotBuilder,
    ) -> None:
        repository, _ = commit_run(artifact_root)
        result = builder.qualify(execution_id=DEFAULT_EXECUTION_ID, artifact_repository=repository)
        assert result.qualified

    def test_qualify_rejected_run_raises(
        self,
        artifact_root: Path,
        builder: ReferenceSnapshotBuilder,
    ) -> None:
        repository, _ = commit_run(artifact_root, include_profile=False)
        with pytest.raises(ReferenceQualificationError):
            builder.qualify(execution_id=DEFAULT_EXECUTION_ID, artifact_repository=repository)
