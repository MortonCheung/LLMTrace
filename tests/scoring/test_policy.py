"""Tests for scoring policy."""

from __future__ import annotations

import pytest

from llmtrace.scoring.errors import InvalidPolicyError
from llmtrace.scoring.models import CapabilityDimension
from llmtrace.scoring.policy import CapabilityScoringPolicy


class TestPolicyCreation:
    """Test policy construction and validation."""

    def test_default_v1_weights_sum_to_one(self) -> None:
        """Default v1 policy weights must sum to exactly 1.0."""
        policy = CapabilityScoringPolicy.create_v1()
        total = sum(policy.dimension_weights.values())
        assert abs(total - 1.0) < 1e-9

    def test_invalid_weights_negative_fails(self) -> None:
        """Negative weight must raise InvalidPolicyError."""
        weights = dict(CapabilityScoringPolicy.create_v1().dimension_weights)
        weights[CapabilityDimension.REASONING] = -0.1
        with pytest.raises(InvalidPolicyError, match="Negative weight"):
            CapabilityScoringPolicy(
                policy_id="test",
                policy_version="1.0",
                dimension_weights=weights,
            )

    def test_invalid_weights_sum_not_one_fails(self) -> None:
        """Weights not summing to 1.0 must raise InvalidPolicyError."""
        weights = {CapabilityDimension.REASONING: 0.5}
        with pytest.raises(ValueError, match="sum to 1.0"):
            CapabilityScoringPolicy(
                policy_id="test",
                policy_version="1.0",
                dimension_weights=weights,
            )

    def test_enabled_dimension_without_weight_fails(self) -> None:
        """Enabled dimension without a weight entry must fail."""
        weights = {CapabilityDimension.REASONING: 1.0}
        with pytest.raises(InvalidPolicyError, match="has no weight"):
            CapabilityScoringPolicy(
                policy_id="test",
                policy_version="1.0",
                dimension_weights=weights,
                enabled_dimensions={
                    CapabilityDimension.REASONING,
                    CapabilityDimension.CODING,
                },
            )

    def test_unknown_dimension_in_weights_fails(self) -> None:
        """All weight keys must be valid CapabilityDimension values."""
        weights = {CapabilityDimension.REASONING: 0.5}
        with pytest.raises(ValueError, match="sum to 1.0"):
            CapabilityScoringPolicy(
                policy_id="test",
                policy_version="1.0",
                dimension_weights=weights,
                enabled_dimensions={CapabilityDimension.REASONING},
            )

    # -- NEW: NaN / Infinity rejection ------------------------------------

    def test_nan_weight_rejected(self) -> None:
        """NaN weight must raise InvalidPolicyError."""
        weights = dict(CapabilityScoringPolicy.create_v1().dimension_weights)
        weights[CapabilityDimension.REASONING] = float("nan")
        with pytest.raises(InvalidPolicyError, match="Non-finite weight"):
            CapabilityScoringPolicy(
                policy_id="test",
                policy_version="1.0",
                dimension_weights=weights,
            )

    def test_infinity_weight_rejected(self) -> None:
        """Infinity weight must raise InvalidPolicyError."""
        weights = dict(CapabilityScoringPolicy.create_v1().dimension_weights)
        weights[CapabilityDimension.REASONING] = float("inf")
        with pytest.raises(InvalidPolicyError, match="Non-finite weight"):
            CapabilityScoringPolicy(
                policy_id="test",
                policy_version="1.0",
                dimension_weights=weights,
            )

    def test_negative_infinity_weight_rejected(self) -> None:
        """Negative infinity weight must raise InvalidPolicyError."""
        weights = dict(CapabilityScoringPolicy.create_v1().dimension_weights)
        weights[CapabilityDimension.REASONING] = float("-inf")
        with pytest.raises(InvalidPolicyError, match="Non-finite weight"):
            CapabilityScoringPolicy(
                policy_id="test",
                policy_version="1.0",
                dimension_weights=weights,
            )

    def test_nan_sum_rejected(self) -> None:
        """NaN total must raise InvalidPolicyError."""
        weights = {CapabilityDimension.REASONING: float("nan")}
        with pytest.raises(InvalidPolicyError, match="Non-finite"):
            CapabilityScoringPolicy(
                policy_id="test",
                policy_version="1.0",
                dimension_weights=weights,
                enabled_dimensions={CapabilityDimension.REASONING},
            )

    # -- NEW: minimum_dimension_coverage validation -----------------------

    def test_negative_minimum_coverage_rejected(self) -> None:
        """Negative minimum_dimension_coverage must fail."""
        with pytest.raises(InvalidPolicyError, match="minimum_dimension_coverage"):
            CapabilityScoringPolicy(
                policy_id="test",
                policy_version="1.0",
                minimum_dimension_coverage=-0.1,
            )

    def test_over_one_minimum_coverage_rejected(self) -> None:
        """minimum_dimension_coverage > 1.0 must fail."""
        with pytest.raises(InvalidPolicyError, match="minimum_dimension_coverage"):
            CapabilityScoringPolicy(
                policy_id="test",
                policy_version="1.0",
                minimum_dimension_coverage=1.1,
            )

    def test_nan_minimum_coverage_rejected(self) -> None:
        """NaN minimum_dimension_coverage must fail."""
        with pytest.raises(InvalidPolicyError, match="finite"):
            CapabilityScoringPolicy(
                policy_id="test",
                policy_version="1.0",
                minimum_dimension_coverage=float("nan"),
            )

    def test_inf_minimum_coverage_rejected(self) -> None:
        """Infinity minimum_dimension_coverage must fail."""
        with pytest.raises(InvalidPolicyError, match="finite"):
            CapabilityScoringPolicy(
                policy_id="test",
                policy_version="1.0",
                minimum_dimension_coverage=float("inf"),
            )


class TestPolicyProperties:
    """Test policy accessor properties."""

    def test_v1_policy_id(self) -> None:
        policy = CapabilityScoringPolicy.create_v1()
        assert policy.policy_id == "llmtrace-capability-v1"
        assert policy.policy_version == "0.1.0"

    def test_v1_enabled_dimensions(self) -> None:
        policy = CapabilityScoringPolicy.create_v1()
        assert CapabilityDimension.REASONING in policy.enabled_dimensions
        assert CapabilityDimension.CODING in policy.enabled_dimensions
        assert CapabilityDimension.MATH_SCIENCE in policy.enabled_dimensions
        assert CapabilityDimension.INSTRUCTION_FOLLOWING in policy.enabled_dimensions
        assert CapabilityDimension.DATA_ANALYSIS not in policy.enabled_dimensions
        assert CapabilityDimension.LONG_CONTEXT not in policy.enabled_dimensions
        assert CapabilityDimension.TOOL_USE not in policy.enabled_dimensions

    def test_v1_calibration_required(self) -> None:
        policy = CapabilityScoringPolicy.create_v1()
        assert policy.calibration_required is True

    def test_coverage_weight(self) -> None:
        policy = CapabilityScoringPolicy.create_v1()
        cw = policy.coverage_weight_for(
            CapabilityDimension.REASONING,
            CapabilityDimension.CODING,
        )
        assert abs(cw - 0.45) < 1e-9

    def test_coverage_weight_full(self) -> None:
        """All seven dimensions sum to 1.0."""
        policy = CapabilityScoringPolicy.create_v1()
        cw = policy.coverage_weight_for(*CapabilityDimension)
        assert abs(cw - 1.0) < 1e-9

    def test_weight_for_individual_dimension(self) -> None:
        policy = CapabilityScoringPolicy.create_v1()
        assert abs(policy.weight_for(CapabilityDimension.REASONING) - 0.25) < 1e-9
        assert abs(policy.weight_for(CapabilityDimension.CODING) - 0.20) < 1e-9
        assert abs(policy.weight_for(CapabilityDimension.MATH_SCIENCE) - 0.15) < 1e-9
        assert abs(policy.weight_for(CapabilityDimension.INSTRUCTION_FOLLOWING) - 0.15) < 1e-9

    def test_dimension_weights_returns_copy(self) -> None:
        """dimension_weights property returns a mutable copy."""
        policy = CapabilityScoringPolicy.create_v1()
        weights = policy.dimension_weights
        weights[CapabilityDimension.REASONING] = 0.99
        # Original policy unchanged
        assert abs(policy.weight_for(CapabilityDimension.REASONING) - 0.25) < 1e-9
