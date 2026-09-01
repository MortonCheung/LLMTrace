"""Scoring-specific exceptions."""

from __future__ import annotations


class ScoringError(Exception):
    """Base exception for all scoring-related errors."""


class InvalidPolicyError(ScoringError):
    """Raised when a scoring policy is invalid (e.g. weight sum != 1.0)."""


class TaskRegistrationError(ScoringError):
    """Raised when a task cannot be registered in the scoring registry."""


class AggregationError(ScoringError):
    """Raised when dimension aggregation fails (e.g. no valid graded tasks)."""


class ReferenceError(ScoringError):
    """Base exception for reference snapshot storage errors."""


class ReferenceNotFoundError(ReferenceError):
    """Raised when a requested reference snapshot_id does not exist."""


class DuplicateSnapshotError(ReferenceError):
    """Raised when saving a reference snapshot_id that already exists.

    Reference snapshots are immutable and append-only — a new measurement
    must use a new snapshot_id rather than overwrite an existing one.
    """


class ReferenceIntegrityError(ReferenceError):
    """Raised when a reference snapshot's on-disk bytes no longer match its recorded hash.

    For a v0.4-A trusted snapshot the recorded hash is the one held in the
    immutable ``<snapshot_id>.manifest.json`` sidecar — never a
    re-serialisation of the object that was just read from disk.
    """

    error_code = "REFERENCE_SNAPSHOT_INTEGRITY_FAILURE"


class ReferenceSnapshotManifestMissingError(ReferenceError):
    """Raised when a trusted ReferenceSet member has no integrity sidecar.

    A v0.3-C legacy snapshot (no ``<snapshot_id>.manifest.json``) remains
    readable for raw capability comparison, but it may never enter a trusted
    ReferenceSet: without an out-of-band anchor there is nothing to verify
    the on-disk bytes against.
    """

    error_code = "REFERENCE_SNAPSHOT_MANIFEST_MISSING"


class ReferenceSnapshotProvenanceMismatchError(ReferenceError):
    """Raised when a snapshot's provenance disagrees with its integrity sidecar.

    The sidecar records the run provenance hashes independently of the
    snapshot body, so editing the body makes the two drift apart — the
    snapshot is then no longer the record the sidecar attests to.
    """

    error_code = "REFERENCE_SNAPSHOT_PROVENANCE_MISMATCH"


class ComparisonError(ScoringError):
    """Base exception for capability comparison errors."""


class SuiteMismatchError(ComparisonError):
    """Raised when reference and candidate were measured on different suites."""


class SuiteVersionMismatchError(ComparisonError):
    """Raised when reference and candidate were measured on different suite versions."""


class ScoringPolicyMismatchError(ComparisonError):
    """Raised when reference and candidate were scored under different scoring policies.

    Suite identity and scoring policy are distinct concepts: two profiles may
    share ``suite_id`` / ``suite_version`` yet still be incomparable because
    their dimension scores were produced under different policies or policy
    versions.  Reusing ``SuiteMismatchError`` would conflate the two.
    """

    error_code = "SCORING_POLICY_MISMATCH"


class IncompatibleCoverageError(ComparisonError):
    """Raised when reference and candidate do not cover the same dimensions.

    Coverage is judged by *comparable* dimensions — a dimension whose
    ``DimensionScoreStatus`` is UNAVAILABLE or INSUFFICIENT_DATA carries no
    measurement and therefore does not count as covered.  A silent partial
    comparison is forbidden.
    """

    error_code = "INCOMPATIBLE_COVERAGE"
