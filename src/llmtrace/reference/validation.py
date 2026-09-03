"""Trusted ReferenceSet validation for formal calibration (v0.4-B).

Why this module exists
----------------------
``ReferenceSet.content_sha256`` is a *canonical self-checksum*: it proves the
file's declared digest agrees with the file's current content, nothing more.
It cannot prove that the file was produced by a trusted builder, that a member
snapshot was swapped, that the sidecar exists, or that the source run is still
the one the set was built from.  In other words::

    ReferenceSet JSON self-hash  !=  trust anchor

v0.4-A established the real anchor (``ReferenceSnapshot`` + its
``<snapshot_id>.manifest.json`` sidecar).  v0.4-B consumes it: a ReferenceSet
may only drive formal 0–100 calibration once every member's trust chain has
been re-verified against the *current* on-disk state, and once the set is
shown to be compatible with the current suite / policy / adapter / generation
context.

Single shared validator
-----------------------
Both the CLI (``--dry-run``) and the runner's ``_preflight()`` call
:func:`validate_reference_set_for_calibration`.  A third, divergent copy of
this logic is exactly how the two paths drifted apart before, so the function
is deliberately local-only and side-effect free:

* reads  — ReferenceSet, snapshot bodies, sidecars, source run manifests
* never  — target HTTP, API key, provider, candidate code execution, artifact write

Consequently a ReferenceSet that cannot support formal calibration is rejected
*before* the first target request is sent, and ``--dry-run`` stays honest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llmtrace.adapters.quick_suite import (
    QUICK_SUITE_ADAPTER_ID,
    QUICK_SUITE_ADAPTER_VERSION,
    QUICK_SUITE_SUITE_ID,
    QUICK_SUITE_SUITE_VERSION,
    get_quick_suite_content_sha256,
    get_quick_suite_generation_config_sha256,
)
from llmtrace.execution.artifacts import RunArtifactRepository
from llmtrace.scoring.calibration import (
    CalibrationError,
    ReferenceCalibrationPolicy,
    ReferenceSetIncompatibleError,
    ReferenceSetIntegrityFailureError,
    UntrustedReferenceSourceError,
)
from llmtrace.scoring.errors import (
    ReferenceIntegrityError,
    ReferenceNotFoundError,
    ReferenceSnapshotManifestMissingError,
    ReferenceSnapshotProvenanceMismatchError,
)
from llmtrace.scoring.policy import CapabilityScoringPolicy
from llmtrace.scoring.reference import ReferenceRepository, ReferenceSnapshotManifest

from .builder import OPERATOR_VERIFIED_API_RUN
from .qualification import ReferenceQualificationPolicy
from .reference_set import ReferenceSet, ReferenceSetIntegrityError, ReferenceSetMember

_SNAPSHOTS_DIRNAME = "snapshots"
_SETS_DIRNAME = "sets"
_CAPABILITY_PROFILE_ARTIFACT = "capability_profile.json"

# Raised by the reference / artifact layers for every way a trust chain can
# break while this module reads it back.
_TRUST_CHAIN_ERRORS = (
    OSError,
    ValueError,
    ReferenceIntegrityError,
    ReferenceNotFoundError,
    ReferenceSnapshotManifestMissingError,
    ReferenceSnapshotProvenanceMismatchError,
)


# ---------------------------------------------------------------------------
# Resolved context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationContext:
    """Immutable, fully verified calibration input for one audit run.

    Only :func:`validate_reference_set_for_calibration` produces it, so a
    context in hand means the trust chain and the compatibility gate both
    passed.
    """

    reference_set: ReferenceSet
    calibration_policy: ReferenceCalibrationPolicy
    scoring_policy: CapabilityScoringPolicy
    snapshot_repository: ReferenceRepository
    reference_root: Path


# ---------------------------------------------------------------------------
# Public validator
# ---------------------------------------------------------------------------


def validate_reference_set_for_calibration(
    *,
    set_path: Path,
    artifact_repository: RunArtifactRepository,
) -> CalibrationContext:
    """Load, verify, and context-check a ReferenceSet for formal calibration.

    Args:
        set_path: Path to a ``ReferenceSet`` JSON file inside a standard
            reference repository (``<root>/sets/<id>_<version>.json``).
        artifact_repository: Repository owning the member source runs; used to
            read the manifest-recorded artifact hashes and the actual
            persisted ``manifest.json`` bytes.

    Raises:
        ReferenceSetIntegrityFailureError: The set is malformed, its self-hash
            does not recompute, the reference repository layout cannot be
            resolved, or any member's trust-chain binding is broken.
        UntrustedReferenceSourceError: A member snapshot's ``source_type`` is
            not ``operator_verified_api_run``.
        ReferenceSetIncompatibleError: The set is trustworthy but was measured
            under a different suite / adapter / scoring / generation /
            qualification context than the current run.
    """
    try:
        return _validate(set_path=set_path, artifact_repository=artifact_repository)
    except CalibrationError:
        raise
    except _TRUST_CHAIN_ERRORS as exc:
        # Any lower-layer failure is a broken trust chain, not a bug to leak.
        raise ReferenceSetIntegrityFailureError(
            f"ReferenceSet at '{set_path}' could not be verified for formal calibration: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _validate(
    *,
    set_path: Path,
    artifact_repository: RunArtifactRepository,
) -> CalibrationContext:
    reference_set = _load_and_verify_set(set_path)
    reference_root = _resolve_reference_root(set_path)
    snapshot_repository = ReferenceRepository.load(reference_root / _SNAPSHOTS_DIRNAME)

    for member in reference_set.members:
        _verify_member(
            member,
            reference_set=reference_set,
            snapshot_repository=snapshot_repository,
            artifact_repository=artifact_repository,
        )

    scoring_policy = CapabilityScoringPolicy.create_v1()
    _assert_compatible(reference_set, scoring_policy)

    return CalibrationContext(
        reference_set=reference_set,
        calibration_policy=ReferenceCalibrationPolicy.create_v1(),
        scoring_policy=scoring_policy,
        snapshot_repository=snapshot_repository,
        reference_root=reference_root,
    )


def _load_and_verify_set(set_path: Path) -> ReferenceSet:
    """Parse the set and re-verify its canonical self-checksum.

    The self-hash is only the first gate: it proves the file is internally
    consistent, never that it is trusted.
    """
    try:
        raw = set_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReferenceSetIntegrityFailureError(f"ReferenceSet unreadable at '{set_path}': {exc}") from exc
    try:
        reference_set = ReferenceSet.model_validate_json(raw)
    except ValueError as exc:
        raise ReferenceSetIntegrityFailureError(
            f"ReferenceSet at '{set_path}' is not a valid ReferenceSet: {exc}"
        ) from exc
    try:
        reference_set.verify_content_hash()
    except ReferenceSetIntegrityError as exc:
        raise ReferenceSetIntegrityFailureError(str(exc)) from exc
    return reference_set


def _resolve_reference_root(set_path: Path) -> Path:
    """Resolve the reference repository root from a set file path.

    Only the existing v0.4-A layout is accepted::

        references/
        ├── snapshots/
        └── sets/<reference_set_id>_<reference_set_version>.json

    A path that cannot be mapped onto that layout fails closed.  Silently
    degrading to "trust the JSON self-hash" would hand an attacker the whole
    trust chain for the price of one recomputed digest.
    """
    sets_dir = set_path.resolve().parent
    if sets_dir.name != _SETS_DIRNAME:
        raise ReferenceSetIntegrityFailureError(
            f"ReferenceSet '{set_path}' is not inside a '{_SETS_DIRNAME}/' directory; formal calibration "
            f"requires the standard layout <reference_root>/{_SETS_DIRNAME}/<set_id>_<set_version>.json"
        )
    root = sets_dir.parent
    snapshots_dir = root / _SNAPSHOTS_DIRNAME
    if not snapshots_dir.is_dir():
        raise ReferenceSetIntegrityFailureError(
            f"reference repository '{root}' has no '{_SNAPSHOTS_DIRNAME}/' directory; the member snapshots' "
            f"trust anchors cannot be verified, so formal calibration is refused"
        )
    return root


def _verify_member(
    member: ReferenceSetMember,
    *,
    reference_set: ReferenceSet,
    snapshot_repository: ReferenceRepository,
    artifact_repository: RunArtifactRepository,
) -> None:
    """Re-verify one member's complete trust chain against the current disk."""
    set_label = f"ReferenceSet '{reference_set.reference_set_id}' v{reference_set.reference_set_version}"

    # -- 6.1 trusted sidecar + 6.2 snapshot SHA binding -------------------
    try:
        actual_snapshot_sha = snapshot_repository.verify_trusted_snapshot(member.snapshot_id)
    except _TRUST_CHAIN_ERRORS as exc:
        raise ReferenceSetIntegrityFailureError(
            f"{set_label} member '{member.snapshot_id}' failed trusted snapshot verification: {exc}"
        ) from exc
    if actual_snapshot_sha != member.snapshot_sha256:
        raise ReferenceSetIntegrityFailureError(
            f"{set_label} member '{member.snapshot_id}' snapshot SHA-256 binding failed: "
            f"verified snapshot bytes {actual_snapshot_sha!r} != member.snapshot_sha256 {member.snapshot_sha256!r}"
        )

    snapshot = snapshot_repository.get(member.snapshot_id)
    provenance = snapshot.provenance
    sidecar = snapshot_repository.load_snapshot_manifest(member.snapshot_id)

    # -- 7. trusted source gate -------------------------------------------
    if provenance.source_type != OPERATOR_VERIFIED_API_RUN:
        raise UntrustedReferenceSourceError(
            f"{set_label} member '{member.snapshot_id}' provenance.source_type is "
            f"{provenance.source_type!r}; formal calibration requires {OPERATOR_VERIFIED_API_RUN!r}"
        )

    if member.execution_id is None:
        raise ReferenceSetIntegrityFailureError(
            f"{set_label} member '{member.snapshot_id}' records no execution_id; "
            f"formal calibration requires a source run artifact"
        )

    # -- 6.3 identity binding ---------------------------------------------
    identity_mismatches = [
        f"{name}: snapshot {actual!r} != member {expected!r}"
        for name, actual, expected in (
            ("snapshot_id", snapshot.snapshot_id, member.snapshot_id),
            ("model_id", snapshot.model_id, member.model_id),
            ("provider_id", snapshot.provider_id, member.provider_id),
            ("execution_id", provenance.execution_id, member.execution_id),
        )
        if actual != expected
    ]
    if identity_mismatches:
        raise ReferenceSetIntegrityFailureError(
            f"{set_label} member '{member.snapshot_id}' identity binding failed: " + "; ".join(identity_mismatches)
        )

    # -- 6.4 / 6.5 source run bindings ------------------------------------
    try:
        manifest = artifact_repository.load_manifest(member.execution_id)
        actual_manifest_sha = artifact_repository.manifest_sha256(member.execution_id)
    except _TRUST_CHAIN_ERRORS as exc:
        raise ReferenceSetIntegrityFailureError(
            f"{set_label} member '{member.snapshot_id}' source run '{member.execution_id}' could not be verified: {exc}"
        ) from exc

    manifest_profile_sha = manifest.artifacts.get(_CAPABILITY_PROFILE_ARTIFACT)
    if manifest_profile_sha is None:
        raise ReferenceSetIntegrityFailureError(
            f"{set_label} member '{member.snapshot_id}' source run '{member.execution_id}' records no "
            f"'{_CAPABILITY_PROFILE_ARTIFACT}' artifact"
        )

    _assert_uniform(
        (
            ("run manifest artifact", manifest_profile_sha),
            ("ReferenceSet member", member.capability_profile_sha256),
            ("snapshot provenance", provenance.capability_profile_sha256),
            ("integrity sidecar", sidecar.capability_profile_sha256),
        ),
        what=f"{set_label} member '{member.snapshot_id}' capability_profile SHA-256 chain",
    )
    _assert_uniform(
        (
            ("persisted manifest bytes", actual_manifest_sha),
            ("snapshot provenance", provenance.run_manifest_sha256),
            ("integrity sidecar", sidecar.run_manifest_sha256),
        ),
        what=f"{set_label} member '{member.snapshot_id}' run manifest SHA-256 chain",
    )
    _assert_uniform(
        (
            ("ReferenceSet member", member.execution_id),
            ("snapshot provenance", provenance.execution_id),
            ("integrity sidecar", sidecar.source_execution_id),
        ),
        what=f"{set_label} member '{member.snapshot_id}' source execution_id chain",
    )


def _assert_uniform(bindings: tuple[tuple[str, str | None], ...], *, what: str) -> None:
    """Require every binding to carry the same non-None value.

    Each list is one provenance fact recorded independently by a different
    writer; a single disagreement means one of the records was rewritten.
    """
    if any(value is None for _label, value in bindings):
        raise ReferenceSetIntegrityFailureError(
            f"{what} is incomplete: " + "; ".join(f"{label}={value!r}" for label, value in bindings)
        )
    if len({value for _label, value in bindings}) != 1:
        raise ReferenceSetIntegrityFailureError(
            f"{what} disagrees: " + "; ".join(f"{label}={value!r}" for label, value in bindings)
        )


def _assert_compatible(reference_set: ReferenceSet, scoring_policy: CapabilityScoringPolicy) -> None:
    """Gate the set against the *current* calibration context.

    A trusted set measured on a different suite, adapter, scoring policy,
    generation config, or qualification policy is not comparable with this
    candidate — version strings alone are not enough, the content SHA is the
    final authority.
    """
    qualification_policy = ReferenceQualificationPolicy.create_v1()
    checks: tuple[tuple[str, str, str], ...] = (
        ("suite_id", reference_set.suite_id, QUICK_SUITE_SUITE_ID),
        ("suite_version", reference_set.suite_version, QUICK_SUITE_SUITE_VERSION),
        ("suite_content_sha256", reference_set.suite_content_sha256, get_quick_suite_content_sha256()),
        ("adapter_id", reference_set.adapter_id, QUICK_SUITE_ADAPTER_ID),
        ("adapter_version", reference_set.adapter_version, QUICK_SUITE_ADAPTER_VERSION),
        ("scoring_policy_id", reference_set.scoring_policy_id, scoring_policy.policy_id),
        ("scoring_policy_version", reference_set.scoring_policy_version, scoring_policy.policy_version),
        (
            "generation_config_sha256",
            reference_set.generation_config_sha256,
            get_quick_suite_generation_config_sha256(),
        ),
        ("qualification_policy_id", reference_set.qualification_policy_id, qualification_policy.policy_id),
        (
            "qualification_policy_version",
            reference_set.qualification_policy_version,
            qualification_policy.policy_version,
        ),
    )
    mismatches = [
        f"{name}: ReferenceSet {actual!r} != current {expected!r}"
        for name, actual, expected in checks
        if actual != expected
    ]
    if mismatches:
        raise ReferenceSetIncompatibleError(
            f"ReferenceSet '{reference_set.reference_set_id}' v{reference_set.reference_set_version} "
            f"is incompatible with the current calibration context: " + "; ".join(mismatches)
        )


__all__: list[str] = [
    "CalibrationContext",
    "validate_reference_set_for_calibration",
    "ReferenceSnapshotManifest",
]
