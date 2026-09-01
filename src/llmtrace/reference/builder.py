"""ReferenceSnapshotBuilder — build immutable snapshots from verified Run Artifacts.

The builder is the *only* production path from a run artifact to a
``ReferenceSnapshot``.  It enforces the strict order (§15):

    verify Run artifacts → qualification passes → build ReferenceSnapshot
    → ReferenceRepository.save_trusted()  (snapshot.json + integrity sidecar)

A snapshot is never built from a transient in-memory profile: the persisted
and verified ``capability_profile.json`` artifact is the single input source
(Gate 3).  Provenance records honest, persisted facts — the manifest's actual
file-bytes SHA, the artifact manifest's recorded profile SHA, and the Quick
Suite's source revisions from its manifest (never a second hardcoded table).
"""

from __future__ import annotations

from datetime import UTC

from llmtrace.adapters.quick_suite import (
    QUICK_SUITE_SUITE_ID,
    QUICK_SUITE_SUITE_VERSION,
    get_quick_suite_source_revisions,
)
from llmtrace.execution.artifacts import RunArtifactRepository
from llmtrace.scoring.models import CapabilityProfile
from llmtrace.scoring.reference import ReferenceProvenance, ReferenceRepository, ReferenceSnapshot

from .qualification import (
    ReferenceQualificationError,
    ReferenceQualificationPolicy,
    ReferenceQualificationResult,
    require_qualified,
)

# The only source_type the production capture path may use (§28).  Tests may
# pass ``test_fixture`` explicitly; the CLI/capture service never exposes that.
OPERATOR_VERIFIED_API_RUN = "operator_verified_api_run"
TEST_FIXTURE = "test_fixture"

_QUALIFICATION_POLICY_ID = "llmtrace_reference_qualification_v1"
_QUALIFICATION_POLICY_VERSION = "0.1.0"


class ReferenceSnapshotBuilder:
    """Construct ``ReferenceSnapshot`` from a qualified, persisted run."""

    def __init__(
        self,
        *,
        qualification_policy: ReferenceQualificationPolicy | None = None,
    ) -> None:
        self._qualification_policy = (
            qualification_policy if qualification_policy is not None else ReferenceQualificationPolicy.create_v1()
        )

    def qualify(
        self,
        *,
        execution_id: str,
        artifact_repository: RunArtifactRepository,
    ) -> ReferenceQualificationResult:
        """Run the full qualification chain; raises ``ReferenceQualificationError`` on REJECT."""
        return require_qualified(
            execution_id=execution_id,
            artifact_repository=artifact_repository,
            policy=self._qualification_policy,
        )

    def build(
        self,
        *,
        execution_id: str,
        artifact_repository: RunArtifactRepository,
        reference_repository: ReferenceRepository,
        provider_id: str,
        snapshot_id: str,
        created_by: str,
        source_type: str = OPERATOR_VERIFIED_API_RUN,
    ) -> ReferenceSnapshot:
        """Verify → qualify → build → save one ``ReferenceSnapshot``.

        Args:
            execution_id: The run artifact to reference.
            artifact_repository: Repository owning the run artifact.
            reference_repository: Append-only store receiving the snapshot.
            provider_id: Operational/vendor label for the reference source
                (metadata, not an identity claim).
            snapshot_id: Unique, filename-safe snapshot identifier.
            created_by: Producer label (operator or tool).
            source_type: Provenance source type.  Production capture uses
                ``operator_verified_api_run``; ``test_fixture`` is only for
                tests and is rejected here for trusted reference sets later.

        Returns:
            The saved ``ReferenceSnapshot``.

        Raises:
            ReferenceQualificationError: If the run fails any gate.
            DuplicateSnapshotError: If ``snapshot_id`` already exists.
        """
        # verify + qualify first (§15) — never build from an unverified run.
        qualification = self.qualify(execution_id=execution_id, artifact_repository=artifact_repository)
        assert qualification.capability_profile is not None

        manifest = artifact_repository.load_manifest(execution_id)

        created_at = manifest.completed_at if manifest.completed_at is not None else manifest.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        created_at = created_at.astimezone(UTC)

        suite_sha256 = manifest.suite_content_sha256
        if suite_sha256 is None:
            # Gate 6 already rejects None; this is a defensive invariant.
            from .qualification import SUITE_CONTENT_MISMATCH

            raise ReferenceQualificationError(
                f"cannot build reference for execution '{execution_id}': suite_content_sha256 is None",
                error_code=SUITE_CONTENT_MISMATCH,
            )

        profile = _verified_profile(qualification, manifest.artifacts.get("capability_profile.json"))

        benchmark_revision = f"{QUICK_SUITE_SUITE_ID}-{QUICK_SUITE_SUITE_VERSION}"

        provenance = ReferenceProvenance(
            source_type=source_type,
            created_by=created_by,
            created_at=created_at,
            suite_sha256=suite_sha256,
            benchmark_revision=benchmark_revision,
            runner_version=_runner_version(),
            execution_id=manifest.execution_id,
            endpoint_redacted=manifest.base_url_redacted,
            adapter_id=manifest.adapter_id,
            adapter_version=manifest.adapter_version,
            generation_config_sha256=manifest.generation_config_sha256,
            run_manifest_sha256=artifact_repository.manifest_sha256(execution_id),
            capability_profile_sha256=manifest.artifacts.get("capability_profile.json"),
            qualification_policy_id=qualification.policy_id,
            qualification_policy_version=qualification.policy_version,
            benchmark_revisions=get_quick_suite_source_revisions(),
        )

        snapshot = ReferenceSnapshot(
            snapshot_id=snapshot_id,
            model_id=manifest.candidate_model_id,
            provider_id=provider_id,
            created_at=created_at,
            suite_id=manifest.suite_id,
            suite_version=manifest.suite_version,
            capability_profile=profile,
            provenance=provenance,
        )

        # save last, after verify + qualify + build (§15).  Trusted snapshots
        # are written with their integrity sidecar, so a later tamper of the
        # persisted bytes is detectable instead of self-verifying.
        return reference_repository.save_trusted(snapshot)


def _verified_profile(
    qualification: ReferenceQualificationResult,
    recorded_profile_sha: str | None,
) -> CapabilityProfile:
    """Return the profile carried by a QUALIFIED result.

    The profile comes from the persisted ``capability_profile.json`` artifact
    (verified against the manifest hash in Gate 1/3) — never a transient
    in-memory object.
    """
    profile = qualification.capability_profile
    if profile is None:
        raise ReferenceQualificationError(
            "qualified result carried no persisted capability profile",
            error_code="MISSING_CAPABILITY_PROFILE",
        )
    return profile


def _runner_version() -> str:
    """LLMTrace runner version recorded in reference provenance."""
    try:
        from importlib.metadata import version

        return version("llmtrace")
    except Exception:
        return "0.0.0"
