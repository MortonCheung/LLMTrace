"""Tests for scoring data models validation."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from llmtrace.scoring.models import (
    CapabilityDimension,
    CapabilityProfile,
    DimensionScoreResult,
    DimensionScoreStatus,
    TaskScoringSpec,
)


class TestTaskScoringSpec:
    """Validate TaskScoringSpec."""

    def test_minimal_spec(self) -> None:
        spec = TaskScoringSpec(
            task_id="test_task",
            dimension=CapabilityDimension.REASONING,
            task_weight=1.0,
        )
        assert spec.task_id == "test_task"
        assert spec.dimension == CapabilityDimension.REASONING
        assert spec.capability_score_eligible is True

    def test_task_weight_zero_fails(self) -> None:
        with pytest.raises(ValueError):
            TaskScoringSpec(
                task_id="zero_task",
                dimension=CapabilityDimension.REASONING,
                task_weight=0.0,
            )

    def test_task_weight_negative_fails(self) -> None:
        with pytest.raises(ValueError):
            TaskScoringSpec(
                task_id="neg_task",
                dimension=CapabilityDimension.REASONING,
                task_weight=-1.0,
            )

    def test_empty_task_id_fails(self) -> None:
        with pytest.raises(ValueError):
            TaskScoringSpec(
                task_id="",
                dimension=CapabilityDimension.REASONING,
                task_weight=1.0,
            )

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValueError):
            TaskScoringSpec(
                task_id="test_task",
                dimension=CapabilityDimension.REASONING,
                task_weight=1.0,
                unknown_field="value",  # type: ignore[call-arg]
            )

    def test_smoke_not_eligible(self) -> None:
        spec = TaskScoringSpec(
            task_id="smoke",
            dimension=CapabilityDimension.REASONING,
            task_weight=1.0,
            capability_score_eligible=False,
        )
        assert spec.capability_score_eligible is False


class TestDimensionScoreResult:
    """Validate DimensionScoreResult."""

    def test_minimal_result(self) -> None:
        result = DimensionScoreResult(
            dimension=CapabilityDimension.REASONING,
            status=DimensionScoreStatus.UNCALIBRATED,
            raw_normalized_score=0.7,
        )
        assert result.calibrated_score is None
        assert result.weighted_contribution == 0.0

    def test_raw_normalized_score_out_of_range_fails(self) -> None:
        with pytest.raises(ValueError):
            DimensionScoreResult(
                dimension=CapabilityDimension.REASONING,
                status=DimensionScoreStatus.UNCALIBRATED,
                raw_normalized_score=1.5,
            )

        with pytest.raises(ValueError):
            DimensionScoreResult(
                dimension=CapabilityDimension.REASONING,
                status=DimensionScoreStatus.UNCALIBRATED,
                raw_normalized_score=-0.1,
            )

    def test_calibrated_score_can_be_none(self) -> None:
        result = DimensionScoreResult(
            dimension=CapabilityDimension.REASONING,
            status=DimensionScoreStatus.UNCALIBRATED,
            raw_normalized_score=0.5,
            calibrated_score=None,
        )
        assert result.calibrated_score is None


class TestCapabilityProfile:
    """Validate CapabilityProfile."""

    def test_minimal_profile(self) -> None:
        profile = CapabilityProfile(
            scoring_policy_id="test-v1",
            scoring_policy_version="0.1.0",
        )
        assert profile.calibrated_total_score is None
        assert profile.provisional_raw_index == 0.0
        assert profile.profile_version == "0.1.0"

    def test_profile_is_frozen(self) -> None:
        """CapabilityProfile must be frozen."""
        profile = CapabilityProfile(
            scoring_policy_id="test-v1",
            scoring_policy_version="0.1.0",
        )
        with pytest.raises((TypeError, ValueError)):  # frozen model raises on mutation
            profile.coverage_weight = 1.0  # type: ignore[misc]

    def test_profile_serialization(self) -> None:
        profile = CapabilityProfile(
            scoring_policy_id="test-v1",
            scoring_policy_version="0.1.0",
            coverage_weight=0.45,
            provisional_raw_index=0.34,
            dimensions=[
                DimensionScoreResult(
                    dimension=CapabilityDimension.REASONING,
                    status=DimensionScoreStatus.UNCALIBRATED,
                    raw_normalized_score=0.8,
                    global_weight=0.25,
                    weighted_contribution=0.20,
                    evidence_refs=[str(uuid4())],
                ),
            ],
            warnings=["test warning"],
        )
        data = json.loads(profile.model_dump_json())
        assert data["profile_version"] == "0.1.0"
        assert data["calibrated_total_score"] is None
        assert data["coverage_weight"] == 0.45
        assert data["provisional_raw_index"] == 0.34
        assert len(data["dimensions"]) == 1
        assert len(data["warnings"]) == 1


class TestCapabilityDimension:
    """Test CapabilityDimension enum."""

    def test_all_seven_dimensions_defined(self) -> None:
        values = {d.value for d in CapabilityDimension}
        assert "reasoning" in values
        assert "coding" in values
        assert "math_science" in values
        assert "instruction_following" in values
        assert "data_analysis" in values
        assert "long_context" in values
        assert "tool_use" in values
        assert len(values) == 7


class TestDimensionScoreStatus:
    """Test DimensionScoreStatus enum."""

    def test_all_statuses_defined(self) -> None:
        values = {s.value for s in DimensionScoreStatus}
        assert "scored" in values
        assert "uncalibrated" in values
        assert "insufficient_data" in values
        assert "unavailable" in values
        assert len(values) == 4
