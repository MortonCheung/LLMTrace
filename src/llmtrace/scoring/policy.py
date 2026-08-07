"""Versioned scoring policy: dimension weights, minimum coverage, and calibration flag."""

from __future__ import annotations

import math

from .errors import InvalidPolicyError
from .models import (
    _LONG_TERM_WEIGHTS,
    CapabilityDimension,
    _validate_weights_sum,
)


def _build_default_dimension_weights() -> dict[CapabilityDimension, float]:
    """Return the default long-term global weights (sum = 1.0)."""
    return dict(_LONG_TERM_WEIGHTS)


class CapabilityScoringPolicy:
    """Versioned, immutable scoring policy.

    Defines global dimension weights, enabled dimensions, minimum coverage,
    and calibration requirements.  Weights must always sum to exactly 1.0
    — the policy must NOT re-normalise enabled dimensions to 100%.

    Typical usage::

        policy = CapabilityScoringPolicy.create_v1()
        assert policy.coverage_weight_for(
            CapabilityDimension.REASONING,
            CapabilityDimension.CODING,
        ) == 0.45
    """

    __slots__ = (
        "_policy_id",
        "_policy_version",
        "_dimension_weights",
        "_enabled_dimensions",
        "_minimum_dimension_coverage",
        "_calibration_required",
        "_description",
    )

    def __init__(
        self,
        *,
        policy_id: str,
        policy_version: str,
        dimension_weights: dict[CapabilityDimension, float] | None = None,
        enabled_dimensions: set[CapabilityDimension] | None = None,
        minimum_dimension_coverage: float = 0.0,
        calibration_required: bool = True,
        description: str = "",
    ) -> None:
        """Create a new scoring policy.

        Args:
            policy_id: Unique policy identifier.
            policy_version: Semantic version string.
            dimension_weights: Global dimension weights. Defaults to long-term weights.
            enabled_dimensions: Currently active dimensions. Defaults to v0.3 set.
            minimum_dimension_coverage: Minimum task_coverage for a dimension to be scored.
            calibration_required: Whether reference calibration is needed for SCORED status.
            description: Human-readable policy description.

        Raises:
            InvalidPolicyError: If weights do not sum to exactly 1.0, or if any
                weight is negative, non-finite, or if an enabled dimension has no weight.
        """
        self._policy_id = policy_id
        self._policy_version = policy_version
        self._calibration_required = calibration_required
        self._description = description

        # Validate minimum_dimension_coverage: [0.0, 1.0] and finite
        if not math.isfinite(minimum_dimension_coverage):
            raise InvalidPolicyError(f"minimum_dimension_coverage must be finite, got {minimum_dimension_coverage}")
        if minimum_dimension_coverage < 0.0 or minimum_dimension_coverage > 1.0:
            raise InvalidPolicyError(
                f"minimum_dimension_coverage must be in [0.0, 1.0], got {minimum_dimension_coverage}"
            )
        self._minimum_dimension_coverage = minimum_dimension_coverage

        # Weights
        weights = dimension_weights if dimension_weights is not None else _build_default_dimension_weights()
        # Validate negatives and non-finite
        for dim, w in weights.items():
            if not math.isfinite(w):
                raise InvalidPolicyError(f"Non-finite weight {w} for dimension {dim.value}")
            if w < 0:
                raise InvalidPolicyError(f"Negative weight {w} for dimension {dim.value}")
        _validate_weights_sum(weights)
        self._dimension_weights = dict(weights)

        # Enabled dimensions
        enabled = enabled_dimensions if enabled_dimensions is not None else _default_enabled_dimensions()
        for dim in enabled:
            if dim not in self._dimension_weights:
                raise InvalidPolicyError(f"Enabled dimension {dim.value} has no weight in the policy")
        self._enabled_dimensions = frozenset(enabled)

    # -- Factory ----------------------------------------------------------

    @classmethod
    def create_v1(cls) -> CapabilityScoringPolicy:
        """Create the v0.3 default scoring policy."""
        return cls(
            policy_id="llmtrace-capability-v1",
            policy_version="0.1.0",
            enabled_dimensions=_default_enabled_dimensions(),
            minimum_dimension_coverage=0.0,
            calibration_required=True,
            description="LLMTrace capability scoring policy v0.1.  "
            "Enabled: reasoning, coding, math_science, instruction_following.  "
            "Remaining dimensions reserved for future activation.",
        )

    # -- Properties -------------------------------------------------------

    @property
    def policy_id(self) -> str:
        return self._policy_id

    @property
    def policy_version(self) -> str:
        return self._policy_version

    @property
    def dimension_weights(self) -> dict[CapabilityDimension, float]:
        """Return a copy of the global dimension weights (sum = 1.0)."""
        return dict(self._dimension_weights)

    @property
    def enabled_dimensions(self) -> frozenset[CapabilityDimension]:
        return self._enabled_dimensions

    @property
    def minimum_dimension_coverage(self) -> float:
        return self._minimum_dimension_coverage

    @property
    def calibration_required(self) -> bool:
        return self._calibration_required

    @property
    def description(self) -> str:
        return self._description

    # -- Helpers ----------------------------------------------------------

    def weight_for(self, dimension: CapabilityDimension) -> float:
        """Return the global weight for a specific dimension."""
        return self._dimension_weights.get(dimension, 0.0)

    def coverage_weight_for(self, *dimensions: CapabilityDimension) -> float:
        """Sum of global weights for the given dimensions.

        Example:
            policy.coverage_weight_for(
                CapabilityDimension.REASONING,
                CapabilityDimension.CODING,
            )
            → 0.25 + 0.20 = 0.45
        """
        return sum(self.weight_for(d) for d in dimensions)

    def is_enabled(self, dimension: CapabilityDimension) -> bool:
        """Check whether a dimension is currently enabled."""
        return dimension in self._enabled_dimensions

    def __repr__(self) -> str:
        return (
            f"CapabilityScoringPolicy(id={self._policy_id!r}, "
            f"v={self._policy_version!r}, "
            f"enabled={sorted(d.value for d in self._enabled_dimensions)})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _default_enabled_dimensions() -> set[CapabilityDimension]:
    """v0.3 default: reasoning, coding, math_science, instruction_following."""
    return {
        CapabilityDimension.REASONING,
        CapabilityDimension.CODING,
        CapabilityDimension.MATH_SCIENCE,
        CapabilityDimension.INSTRUCTION_FOLLOWING,
    }
