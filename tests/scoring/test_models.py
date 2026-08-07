"""Tests for scoring data models."""

from __future__ import annotations

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
    """Tests for TaskScoringSpec."""

    def test_spec_normal_creation(self) -> None:
        spec = TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0)
        assert spec.task_id == "task_a"
        assert spec.task_weight == 1.0
        assert spec.capability_score_eligible is True

    def test_spec_zero_weight_fails(self) -> None:
        with pytest.raises(ValueError):
            TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=0.0)

    def test_spec_negative_weight_fails(self) -> None:
        with pytest.raises(ValueError):
            TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=-1.0)

    def test_spec_empty_task_id_fails(self) -> None:
        with pytest.raises(ValueError):
            TaskScoringSpec(task_id="", dimension=CapabilityDimension.REASONING, task_weight=1.0)

    def test_spec_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValueError):
            TaskScoringSpec(
                task_id="task_a",
                dimension=CapabilityDimension.REASONING,
                task_weight=1.0,
                unknown_field=123,  # type: ignore[call-arg]
            )

    def test_spec_smoke_eligible(self) -> None:
        spec = TaskScoringSpec(
            task_id="smoke",
            dimension=CapabilityDimension.REASONING,
            task_weight=1.0,
            capability_score_eligible=False,
        )
        assert spec.capability_score_eligible is False

    def test_spec_is_frozen(self) -> None:
        """TaskScoringSpec must be frozen."""
        spec = TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0)
        with pytest.raises((TypeError, ValueError)):
            spec.task_weight = 999  # type: ignore[misc]


class TestDimensionScoreResult:
    """Tests for DimensionScoreResult."""

    def test_score_range_zero_to_one(self) -> None:
        result = DimensionScoreResult(
            dimension=CapabilityDimension.REASONING,
            status=DimensionScoreStatus.UNCALIBRATED,
            raw_normalized_score=0.5,
        )
        assert result.raw_normalized_score == 0.5

    def test_score_out_of_range_fails(self) -> None:
        with pytest.raises(ValueError):
            DimensionScoreResult(
                dimension=CapabilityDimension.REASONING,
                status=DimensionScoreStatus.UNCALIBRATED,
                raw_normalized_score=1.5,
            )

    def test_coverage_out_of_range_fails(self) -> None:
        with pytest.raises(ValueError):
            DimensionScoreResult(
                dimension=CapabilityDimension.REASONING,
                status=DimensionScoreStatus.UNCALIBRATED,
                raw_normalized_score=0.5,
                task_coverage=1.5,
            )

    def test_evidence_refs_default_empty(self) -> None:
        result = DimensionScoreResult(
            dimension=CapabilityDimension.REASONING,
            status=DimensionScoreStatus.UNCALIBRATED,
            raw_normalized_score=0.5,
        )
        assert result.evidence_refs == ()

    def test_source_task_ids_default_empty(self) -> None:
        result = DimensionScoreResult(
            dimension=CapabilityDimension.REASONING,
            status=DimensionScoreStatus.UNCALIBRATED,
            raw_normalized_score=0.5,
        )
        assert result.source_task_ids == ()

    def test_warnings_default_empty(self) -> None:
        result = DimensionScoreResult(
            dimension=CapabilityDimension.REASONING,
            status=DimensionScoreStatus.UNCALIBRATED,
            raw_normalized_score=0.5,
        )
        assert result.warnings == ()

    def test_calibrated_score_none_valid(self) -> None:
        result = DimensionScoreResult(
            dimension=CapabilityDimension.REASONING,
            status=DimensionScoreStatus.UNCALIBRATED,
            raw_normalized_score=0.5,
            calibrated_score=None,
        )
        assert result.calibrated_score is None

    def test_calibrated_score_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be in \\[0, 100\\]"):
            DimensionScoreResult(
                dimension=CapabilityDimension.REASONING,
                status=DimensionScoreStatus.SCORED,
                raw_normalized_score=0.5,
                calibrated_score=101.0,
            )

    def test_is_frozen(self) -> None:
        result = DimensionScoreResult(
            dimension=CapabilityDimension.REASONING,
            status=DimensionScoreStatus.UNCALIBRATED,
            raw_normalized_score=0.5,
        )
        with pytest.raises((TypeError, ValueError)):
            result.raw_normalized_score = 0.9  # type: ignore[misc]

    def test_invalid_evidence_uuid_rejected(self) -> None:
        """Invalid UUID in evidence_refs → ValueError."""
        with pytest.raises(ValueError):
            DimensionScoreResult(
                dimension=CapabilityDimension.REASONING,
                status=DimensionScoreStatus.UNCALIBRATED,
                raw_normalized_score=0.5,
                evidence_refs=("not-a-uuid",),
            )

    def test_evidence_refs_deduplicated(self) -> None:
        """Duplicate evidence UUIDs are deduplicated on construction."""
        eid = str(uuid4())
        result = DimensionScoreResult(
            dimension=CapabilityDimension.REASONING,
            status=DimensionScoreStatus.UNCALIBRATED,
            raw_normalized_score=0.5,
            evidence_refs=(eid, eid),
        )
        assert result.evidence_refs.count(eid) == 1
        assert len(result.evidence_refs) == 1

    def test_evidence_refs_preserves_order(self) -> None:
        """First-seen order of evidence UUIDs is preserved."""
        e1, e2 = str(uuid4()), str(uuid4())
        result = DimensionScoreResult(
            dimension=CapabilityDimension.REASONING,
            status=DimensionScoreStatus.UNCALIBRATED,
            raw_normalized_score=0.5,
            evidence_refs=(e1, e2, e1),  # e1 duplicated, first occurrence wins
        )
        assert result.evidence_refs == (e1, e2)


class TestCapabilityProfile:
    """Tests for CapabilityProfile."""

    def test_capability_dimension_enum(self) -> None:
        assert CapabilityDimension.REASONING.value == "reasoning"
        assert CapabilityDimension.CODING.value == "coding"
        assert len(CapabilityDimension) == 7

    def test_dimension_score_status_enum(self) -> None:
        assert set(DimensionScoreStatus) == {
            DimensionScoreStatus.SCORED,
            DimensionScoreStatus.UNCALIBRATED,
            DimensionScoreStatus.INSUFFICIENT_DATA,
            DimensionScoreStatus.UNAVAILABLE,
        }

    def test_is_frozen(self) -> None:
        eid = str(uuid4())
        profile = CapabilityProfile(
            scoring_policy_id="test-policy",
            scoring_policy_version="1.0",
            evidence_refs=(eid,),
        )
        with pytest.raises((TypeError, ValueError)):
            profile.coverage_weight = 1.0  # type: ignore[misc]

    def test_calibrated_total_score_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be in \\[0, 100\\]"):
            CapabilityProfile(
                scoring_policy_id="test-policy",
                scoring_policy_version="1.0",
                calibrated_total_score=101.0,
            )

    def test_invalid_evidence_uuid_rejected(self) -> None:
        """Evidence refs that are not valid UUIDs must be rejected."""
        with pytest.raises(ValueError):
            CapabilityProfile(
                scoring_policy_id="test-policy",
                scoring_policy_version="1.0",
                evidence_refs=("not-a-uuid",),  # type: ignore[arg-type]
            )

    def test_valid_evidence_uuid_accepted(self) -> None:
        eid = str(uuid4())
        profile = CapabilityProfile(
            scoring_policy_id="test-policy",
            scoring_policy_version="1.0",
            evidence_refs=(eid,),
        )
        assert eid in profile.evidence_refs

    def test_evidence_refs_is_tuple(self) -> None:
        eid = str(uuid4())
        profile = CapabilityProfile(
            scoring_policy_id="test-policy",
            scoring_policy_version="1.0",
            evidence_refs=(eid,),
        )
        assert isinstance(profile.evidence_refs, tuple)

    def test_evidence_refs_deduplicated(self) -> None:
        """Duplicate evidence UUIDs are deduplicated."""
        eid = str(uuid4())
        profile = CapabilityProfile(
            scoring_policy_id="test-policy",
            scoring_policy_version="1.0",
            evidence_refs=(eid, eid),
        )
        assert len(profile.evidence_refs) == 1

    def test_evidence_refs_order_preserved(self) -> None:
        """First-seen order of evidence UUIDs is preserved."""
        e1, e2 = str(uuid4()), str(uuid4())
        profile = CapabilityProfile(
            scoring_policy_id="test-policy",
            scoring_policy_version="1.0",
            evidence_refs=(e1, e2, e1),
        )
        assert profile.evidence_refs == (e1, e2)
