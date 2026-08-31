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


class ComparisonError(ScoringError):
    """Base exception for capability comparison errors."""


class SuiteMismatchError(ComparisonError):
    """Raised when reference and candidate were measured on different suites."""


class SuiteVersionMismatchError(ComparisonError):
    """Raised when reference and candidate were measured on different suite versions."""


class IncompatibleCoverageError(ComparisonError):
    """Raised when reference and candidate do not cover the same dimensions.

    Dimensions must match exactly — a silent partial comparison is forbidden.
    """

    error_code = "INCOMPATIBLE_COVERAGE"
