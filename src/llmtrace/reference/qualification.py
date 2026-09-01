"""Reference qualification policy and gate chain (v0.4-A).

A run artifact may become a Trusted Reference only if it passes every
qualification gate, executed in a clear order and failing closed:

    Gate 1  — Artifact Integrity        (RunArtifactRepository.verify)
    Gate 2  — Capability Profile exists (manifest + file)
    Gate 3  — Use the persisted profile (never a transient in-memory object)
    Gate 4  — Measurement complete      (32/32 GRADED, 0 failure, 0 ungradable)
    Gate 5  — Scoring Policy            (plan / profile / manifest identical)
    Gate 6  — Suite                     (id / version / content SHA present + matching)
    Gate 7  — Generation Config         (plan / manifest identical)
    Gate 8  — Adapter                   (id + version recorded)
    Gate 9  — Capability Coverage       (computed from policy, never hardcoded)
    Gate 10 — Dimension Coverage        (comparable set == policy enabled set)

Any failure REJECTS the run; there is no "save anyway + warning" path.
"""

from __future__ import annotations

import hashlib
import json
import math

from pydantic import BaseModel, Field, ValidationError

from llmtrace.adapters.quick_suite import (
    QUICK_SUITE_BENCHMARK_REQUESTS,
    QUICK_SUITE_SUITE_ID,
    QUICK_SUITE_SUITE_VERSION,
    get_quick_suite_content_sha256,
    get_quick_suite_generation_config,
)
from llmtrace.benchmarks.models import BenchmarkRunResult, ItemStatus
from llmtrace.execution.artifacts import ArtifactIntegrityError, ArtifactNotFoundError, RunArtifactRepository
from llmtrace.scoring.models import CapabilityProfile, DimensionScoreStatus
from llmtrace.scoring.policy import CapabilityScoringPolicy

from .models import ReferenceQualificationResult, ReferenceQualificationStatus

# ---------------------------------------------------------------------------
# Machine-readable reason codes (§13)
# ---------------------------------------------------------------------------

ARTIFACT_INTEGRITY_FAILURE = "ARTIFACT_INTEGRITY_FAILURE"
MISSING_CAPABILITY_PROFILE = "MISSING_CAPABILITY_PROFILE"
INCOMPLETE_MEASUREMENT = "INCOMPLETE_MEASUREMENT"
BENCHMARK_FAILURE_PRESENT = "BENCHMARK_FAILURE_PRESENT"
UNGRADABLE_ITEM_PRESENT = "UNGRADABLE_ITEM_PRESENT"
SUITE_MISMATCH = "SUITE_MISMATCH"
SUITE_CONTENT_MISMATCH = "SUITE_CONTENT_MISMATCH"
SCORING_POLICY_MISMATCH = "SCORING_POLICY_MISMATCH"
GENERATION_CONFIG_MISMATCH = "GENERATION_CONFIG_MISMATCH"
ADAPTER_MISMATCH = "ADAPTER_MISMATCH"
INCOMPATIBLE_COVERAGE = "INCOMPATIBLE_COVERAGE"
MISSING_PROVENANCE = "MISSING_PROVENANCE"


class ReferenceQualificationError(ReferenceError):
    """Raised when qualification is invoked and the run is REJECTED.

    Carries the machine-readable ``error_code`` so callers can branch
    programmatically instead of parsing free-form text.
    """

    error_code = "REFERENCE_QUALIFICATION_FAILED"

    def __init__(self, message: str, *, error_code: str | None = None) -> None:
        super().__init__(message)
        if error_code is not None:
            self.error_code = error_code


class ReferenceQualificationPolicy(BaseModel):
    """Versioned qualification policy — the rules, not scattered ``if``s."""

    policy_id: str = Field(..., min_length=1)
    policy_version: str = Field(..., min_length=1)
    description: str = Field(default="", description="Human-readable policy description")

    model_config = {"frozen": True, "extra": "forbid"}

    @classmethod
    def create_v1(cls) -> ReferenceQualificationPolicy:
        """Create the v0.4-A qualification policy."""
        return cls(
            policy_id="llmtrace_reference_qualification_v1",
            policy_version="0.1.0",
            description=(
                "v0.4-A reference qualification: artifact integrity, persisted profile, "
                "complete measurement (32/32 graded), suite/content/generation/adapter/score "
                "consistency, and full dimension coverage. Fail closed on any mismatch."
            ),
        )


# ---------------------------------------------------------------------------
# Qualification
# ---------------------------------------------------------------------------


def _expected_generation_config_sha256() -> str:
    """Canonical SHA-256 of the Quick Suite generation config (single source)."""
    config = get_quick_suite_generation_config()
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_benchmark_runs(
    artifact_repository: RunArtifactRepository, execution_id: str
) -> list[BenchmarkRunResult] | None:
    """Parse ``benchmark_runs.json`` from the run; None if the artifact is absent.

    The artifact's bytes are verified by the repository's integrity check
    (Gate 1) before this parse — a tampered file raises earlier.
    """
    try:
        raw = artifact_repository.read_artifact(execution_id, "benchmark_runs.json")
    except ArtifactNotFoundError:
        return None
    data = json.loads(raw)
    runs = data.get("runs", []) if isinstance(data, dict) else data
    return [BenchmarkRunResult.model_validate(r) for r in runs]


def _measurement_counts(runs: list[BenchmarkRunResult]) -> tuple[int, int, int]:
    """Return ``(graded, failure, ungradable)`` across every item result.

    GRADED items with score 0.0 are still *valid* measurements — a wrong
    answer is a measured answer, never a failure.  Only FAILURE / UNGRADABLE
    statuses represent lost measurement.
    """
    graded = failure = ungradable = 0
    for run in runs:
        for attempt in run.task_attempts:
            for item in attempt.item_results:
                if item.status == ItemStatus.GRADED:
                    graded += 1
                elif item.status == ItemStatus.FAILURE:
                    failure += 1
                else:
                    ungradable += 1
    return graded, failure, ungradable


def _rejected(
    policy: ReferenceQualificationPolicy,
    execution_id: str,
    reason_codes: list[str],
    warnings: list[str],
) -> ReferenceQualificationResult:
    return ReferenceQualificationResult(
        status=ReferenceQualificationStatus.REJECTED,
        policy_id=policy.policy_id,
        policy_version=policy.policy_version,
        execution_id=execution_id,
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        warnings=tuple(warnings),
    )


def qualify_reference_run(
    *,
    execution_id: str,
    artifact_repository: RunArtifactRepository,
    policy: ReferenceQualificationPolicy | None = None,
) -> ReferenceQualificationResult:
    """Run the qualification gate chain over one persisted run artifact.

    Every gate fails closed: any violation produces a REJECTED result with
    machine-readable ``reason_codes``.  On QUALIFIED the result carries the
    *persisted and verified* capability profile (Gate 3).

    Only ``create_v1()`` is accepted.  An unknown policy id/version is
    rejected because v0.4-A hardcodes gate rules — an arbitrary label
    would produce false provenance.

    Args:
        execution_id: RunArtifact execution to qualify.
        artifact_repository: Repository owning the run artifact.
        policy: Qualification policy; defaults to ``create_v1()``.
    """
    resolved_policy = policy if policy is not None else ReferenceQualificationPolicy.create_v1()

    _expected_v1 = ReferenceQualificationPolicy.create_v1()
    if (
        resolved_policy.policy_id != _expected_v1.policy_id
        or resolved_policy.policy_version != _expected_v1.policy_version
    ):
        return _rejected(
            resolved_policy,
            execution_id,
            ["UNKNOWN_POLICY"],
            [
                f"qualification policy '{resolved_policy.policy_id}@{resolved_policy.policy_version}' "
                f"is not the accepted v1 policy; trusted provenance requires create_v1()",
            ],
        )
    reason_codes: list[str] = []
    warnings: list[str] = []

    # -- Gate 1: Artifact Integrity -----------------------------------------
    # Fail closed on anything that makes the artifact untrustworthy — missing,
    # hash-mismatched, unreadable, unparseable, or schema-invalid.  A
    # malformed manifest must never escape as an exception: the caller is
    # entitled to a structured ReferenceQualificationResult.
    try:
        artifact_repository.verify(execution_id)
    except (ArtifactIntegrityError, ArtifactNotFoundError) as exc:
        warnings.append(str(exc))
        return _rejected(resolved_policy, execution_id, [ARTIFACT_INTEGRITY_FAILURE], warnings)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        warnings.append(f"manifest of execution '{execution_id}' is unreadable or malformed: {exc}")
        return _rejected(resolved_policy, execution_id, [ARTIFACT_INTEGRITY_FAILURE], warnings)

    try:
        manifest = artifact_repository.load_manifest(execution_id)
    except (ArtifactNotFoundError, ArtifactIntegrityError, OSError, json.JSONDecodeError, ValidationError) as exc:
        warnings.append(f"manifest of execution '{execution_id}' is unreadable or malformed: {exc}")
        return _rejected(resolved_policy, execution_id, [ARTIFACT_INTEGRITY_FAILURE], warnings)

    # -- Gate 2: Capability Profile exists ----------------------------------
    profile = None
    recorded_profile_sha = manifest.artifacts.get("capability_profile.json")
    if recorded_profile_sha is None:
        reason_codes.append(MISSING_CAPABILITY_PROFILE)
    else:
        # -- Gate 3: use the persisted profile (never the transient object) --
        try:
            raw_profile = artifact_repository.read_artifact(execution_id, "capability_profile.json")
        except ArtifactNotFoundError:
            reason_codes.append(MISSING_CAPABILITY_PROFILE)
        else:
            try:
                profile = CapabilityProfile.model_validate_json(raw_profile)
            except ValueError:
                reason_codes.append(ARTIFACT_INTEGRITY_FAILURE)

    # -- Gate 4: Measurement complete ---------------------------------------
    runs = _load_benchmark_runs(artifact_repository, execution_id)
    if runs is None:
        reason_codes.append(INCOMPLETE_MEASUREMENT)
    else:
        graded, failure, ungradable = _measurement_counts(runs)
        total = graded + failure + ungradable
        if total != QUICK_SUITE_BENCHMARK_REQUESTS or graded != total:
            reason_codes.append(INCOMPLETE_MEASUREMENT)
        if failure:
            reason_codes.append(BENCHMARK_FAILURE_PRESENT)
        if ungradable:
            reason_codes.append(UNGRADABLE_ITEM_PRESENT)

    # -- Gate 5: Scoring Policy (plan / profile / manifest identical) --------
    scoring_policy = CapabilityScoringPolicy.create_v1()
    manifest_policy_ok = (
        manifest.scoring_policy_id == scoring_policy.policy_id
        and manifest.scoring_policy_version == scoring_policy.policy_version
    )
    profile_policy_ok = profile is None or (
        profile.scoring_policy_id == scoring_policy.policy_id
        and profile.scoring_policy_version == scoring_policy.policy_version
    )
    if not manifest_policy_ok or not profile_policy_ok:
        reason_codes.append(SCORING_POLICY_MISMATCH)

    # -- Gate 6: Suite -------------------------------------------------------
    if manifest.suite_id != QUICK_SUITE_SUITE_ID or manifest.suite_version != QUICK_SUITE_SUITE_VERSION:
        reason_codes.append(SUITE_MISMATCH)
    if manifest.suite_content_sha256 is None:
        # Old pre-v0.4-A run: no content identity → never an automatic reference.
        reason_codes.append(SUITE_CONTENT_MISMATCH)
    else:
        try:
            expected_suite_sha = get_quick_suite_content_sha256()
        except Exception:
            reason_codes.append(SUITE_CONTENT_MISMATCH)
        else:
            if manifest.suite_content_sha256 != expected_suite_sha:
                reason_codes.append(SUITE_CONTENT_MISMATCH)

    # -- Gate 7: Generation Config -------------------------------------------
    if manifest.generation_config_sha256 != _expected_generation_config_sha256():
        reason_codes.append(GENERATION_CONFIG_MISMATCH)

    # -- Gate 8: Adapter -----------------------------------------------------
    if not manifest.adapter_id or not manifest.adapter_version:
        reason_codes.append(ADAPTER_MISMATCH)

    # -- Gate 9: Capability Coverage (computed from policy, never hardcoded) --
    if profile is not None:
        expected_coverage = scoring_policy.coverage_weight_for(*scoring_policy.enabled_dimensions)
        if not math.isclose(profile.coverage_weight, expected_coverage, rel_tol=0.0, abs_tol=1e-9):
            reason_codes.append(INCOMPATIBLE_COVERAGE)

    # -- Gate 10: Dimension Coverage -----------------------------------------
    if profile is not None:
        # The comparable set must be *exactly* the policy's enabled set.  A
        # superset is just as incompatible as a subset: an extra SCORED /
        # UNCALIBRATED dimension changes what the profile measures, so a run
        # that scored dimensions the policy disabled is a different
        # measurement, not a better one.
        actual_comparable_dimensions = {
            result.dimension
            for result in profile.dimensions
            if result.status in (DimensionScoreStatus.SCORED, DimensionScoreStatus.UNCALIBRATED)
        }
        expected_dimensions = set(scoring_policy.enabled_dimensions)
        if actual_comparable_dimensions != expected_dimensions:
            reason_codes.append(INCOMPATIBLE_COVERAGE)

        # Defence in depth: an enabled dimension that is present but carries no
        # measurement must still be rejected, even though the set comparison
        # above already covers it (a status downgrade shrinks the comparable
        # set).  Kept so a future status change cannot silently widen Gate 10.
        profile_dims = {d.dimension: d for d in profile.dimensions}
        for dim in scoring_policy.enabled_dimensions:
            dim_result = profile_dims.get(dim)
            if dim_result is None:
                reason_codes.append(INCOMPATIBLE_COVERAGE)
                break
            if dim_result.status in (DimensionScoreStatus.UNAVAILABLE, DimensionScoreStatus.INSUFFICIENT_DATA):
                reason_codes.append(INCOMPATIBLE_COVERAGE)
                break

    if reason_codes:
        return _rejected(resolved_policy, execution_id, reason_codes, warnings)

    return ReferenceQualificationResult(
        status=ReferenceQualificationStatus.QUALIFIED,
        policy_id=resolved_policy.policy_id,
        policy_version=resolved_policy.policy_version,
        execution_id=execution_id,
        capability_profile=profile,
        warnings=tuple(warnings),
    )


def require_qualified(
    *,
    execution_id: str,
    artifact_repository: RunArtifactRepository,
    policy: ReferenceQualificationPolicy | None = None,
) -> ReferenceQualificationResult:
    """Like :func:`qualify_reference_run` but raises on REJECTED.

    Raises:
        ReferenceQualificationError: When the run is REJECTED, carrying the
            first machine-readable reason code as ``error_code``.
    """
    result = qualify_reference_run(execution_id=execution_id, artifact_repository=artifact_repository, policy=policy)
    if not result.qualified:
        first_code = result.reason_codes[0] if result.reason_codes else "REFERENCE_QUALIFICATION_FAILED"
        raise ReferenceQualificationError(
            f"reference qualification rejected run '{execution_id}' with {list(result.reason_codes)}",
            error_code=first_code,
        )
    return result


# Re-export the result model for a single import point.
__all__: list[str] = [
    "ReferenceQualificationResult",
    "ReferenceQualificationStatus",
    "ReferenceQualificationPolicy",
    "ReferenceQualificationError",
    "qualify_reference_run",
    "require_qualified",
    "ARTIFACT_INTEGRITY_FAILURE",
    "MISSING_CAPABILITY_PROFILE",
    "INCOMPLETE_MEASUREMENT",
    "BENCHMARK_FAILURE_PRESENT",
    "UNGRADABLE_ITEM_PRESENT",
    "SUITE_MISMATCH",
    "SUITE_CONTENT_MISMATCH",
    "SCORING_POLICY_MISMATCH",
    "GENERATION_CONFIG_MISMATCH",
    "ADAPTER_MISMATCH",
    "INCOMPATIBLE_COVERAGE",
    "MISSING_PROVENANCE",
]
