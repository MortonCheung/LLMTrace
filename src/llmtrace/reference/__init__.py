"""v0.4-A Trusted Reference foundation.

The reference package owns the trusted, immutable, append-only pipeline
from a qualified Run Artifact to a ``ReferenceSnapshot`` and on to a
self-checksummed ``ReferenceSet``:

    Run Artifact → qualification (Gate 1–10) → ReferenceSnapshotBuilder
        → ReferenceRepository.save_trusted (snapshot.json + integrity sidecar)
        → ReferenceSetBuilder (12-gate) → ReferenceSetRepository

Security invariants (§34–§35): the reference layer stores only hashes,
provenance, profiles, and immutable pointers.  It never stores response
text, prompts, raw answers, full HTTP responses, or API keys, and endpoints
appear only in redacted form.
"""

from __future__ import annotations

from .builder import OPERATOR_VERIFIED_API_RUN, TEST_FIXTURE, ReferenceSnapshotBuilder
from .capture import ReferenceCaptureResult, ReferenceCaptureService, ReferenceCaptureStatus
from .models import ReferenceQualificationResult, ReferenceQualificationStatus
from .qualification import (
    ADAPTER_MISMATCH,
    ARTIFACT_INTEGRITY_FAILURE,
    BENCHMARK_FAILURE_PRESENT,
    GENERATION_CONFIG_MISMATCH,
    INCOMPATIBLE_COVERAGE,
    INCOMPLETE_MEASUREMENT,
    MISSING_CAPABILITY_PROFILE,
    MISSING_PROVENANCE,
    SCORING_POLICY_MISMATCH,
    SUITE_CONTENT_MISMATCH,
    SUITE_MISMATCH,
    UNGRADABLE_ITEM_PRESENT,
    ReferenceQualificationError,
    ReferenceQualificationPolicy,
    qualify_reference_run,
    require_qualified,
)
from .reference_set import (
    DuplicateReferenceSetError,
    FixtureReferenceError,
    ReferenceSet,
    ReferenceSetBuilder,
    ReferenceSetCompatibilityError,
    ReferenceSetError,
    ReferenceSetIntegrityError,
    ReferenceSetMember,
    ReferenceSetNotFoundError,
)
from .repository import ReferenceSetRepository
from .validation import CalibrationContext, validate_reference_set_for_calibration

__all__: list[str] = [
    # builder
    "OPERATOR_VERIFIED_API_RUN",
    "TEST_FIXTURE",
    "ReferenceSnapshotBuilder",
    # capture
    "ReferenceCaptureResult",
    "ReferenceCaptureService",
    "ReferenceCaptureStatus",
    # qualification models
    "ReferenceQualificationResult",
    "ReferenceQualificationStatus",
    # qualification
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
    "ReferenceQualificationError",
    "ReferenceQualificationPolicy",
    "qualify_reference_run",
    "require_qualified",
    # reference set
    "ReferenceSetError",
    "DuplicateReferenceSetError",
    "ReferenceSetNotFoundError",
    "ReferenceSetIntegrityError",
    "ReferenceSetCompatibilityError",
    "FixtureReferenceError",
    "ReferenceSetMember",
    "ReferenceSet",
    "ReferenceSetBuilder",
    # repository
    "ReferenceSetRepository",
    # calibration validation
    "CalibrationContext",
    "validate_reference_set_for_calibration",
]
