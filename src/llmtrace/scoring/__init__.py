"""LLMTrace capability scoring engine.

Independent scoring layer that consumes BenchmarkRunResult directly
(not reporting models) and produces per-dimension scores and an
immutable CapabilityProfile.

Architecture:
    benchmarks
        ↓
    scoring          ← this package
        ↓
    reporting
"""

from __future__ import annotations

from llmtrace.scoring.aggregator import (
    TaskScoringRegistry,
    aggregate_capability_profile,
    aggregate_dimension_score,
)
from llmtrace.scoring.comparison import (
    COMPARABLE_STATUSES,
    CapabilityComparator,
    ComparisonResult,
    DimensionDiff,
    comparable_dimensions,
)
from llmtrace.scoring.errors import (
    AggregationError,
    ComparisonError,
    DuplicateSnapshotError,
    IncompatibleCoverageError,
    InvalidPolicyError,
    ReferenceError,
    ReferenceIntegrityError,
    ReferenceNotFoundError,
    ReferenceSnapshotManifestMissingError,
    ReferenceSnapshotProvenanceMismatchError,
    ScoringPolicyMismatchError,
    SuiteMismatchError,
    SuiteVersionMismatchError,
    TaskRegistrationError,
)
from llmtrace.scoring.models import (
    CapabilityDimension,
    CapabilityProfile,
    DimensionScoreResult,
    DimensionScoreStatus,
    TaskScoringSpec,
)
from llmtrace.scoring.policy import CapabilityScoringPolicy
from llmtrace.scoring.reference import (
    MANIFEST_VERSION_V1,
    ReferenceProvenance,
    ReferenceRepository,
    ReferenceSnapshot,
    ReferenceSnapshotManifest,
)

__all__ = [
    "CapabilityDimension",
    "CapabilityProfile",
    "CapabilityScoringPolicy",
    "DimensionScoreResult",
    "DimensionScoreStatus",
    "TaskScoringRegistry",
    "TaskScoringSpec",
    "aggregate_capability_profile",
    "aggregate_dimension_score",
    "CapabilityComparator",
    "ComparisonResult",
    "DimensionDiff",
    "COMPARABLE_STATUSES",
    "comparable_dimensions",
    "ReferenceProvenance",
    "ReferenceRepository",
    "ReferenceSnapshot",
    "ReferenceSnapshotManifest",
    "MANIFEST_VERSION_V1",
    "AggregationError",
    "ComparisonError",
    "DuplicateSnapshotError",
    "IncompatibleCoverageError",
    "InvalidPolicyError",
    "ReferenceError",
    "ReferenceIntegrityError",
    "ReferenceNotFoundError",
    "ReferenceSnapshotManifestMissingError",
    "ReferenceSnapshotProvenanceMismatchError",
    "ScoringPolicyMismatchError",
    "SuiteMismatchError",
    "SuiteVersionMismatchError",
    "TaskRegistrationError",
]
