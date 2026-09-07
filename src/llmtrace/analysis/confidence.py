"""Confidence v1 — an experimental, deterministic trust label for one measurement.

Confidence is *not* a statistical CI (that stays in a later phase; §36).  It
is a versioned rule over already-verified facts — measurement health,
calibration availability, and reference identity count — so the UI can label
a 0–100 capability score with how much it may be trusted.

Labels: ``High`` / ``Medium`` / ``Low`` / ``Unavailable`` (§34–§35).
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from llmtrace.execution.models import BenchmarkMeasurementSummary
    from llmtrace.scoring.models import CapabilityProfile


class ConfidenceLevel(StrEnum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    UNAVAILABLE = "Unavailable"


class ConfidenceAssessment(BaseModel):
    """One confidence verdict plus the human-readable reasons behind it."""

    level: ConfidenceLevel = Field(...)
    policy_id: str = Field(..., min_length=1)
    policy_version: str = Field(..., min_length=1)
    measurement_available: bool = Field(..., description="A graded measurement exists")
    calibrated: bool = Field(..., description="Profile carries formal reference calibration")
    reference_identity_count: int | None = Field(default=None, ge=1)
    total_item_count: int = Field(default=0, ge=0)
    graded_item_count: int = Field(default=0, ge=0)
    failure_item_count: int = Field(default=0, ge=0)
    reasons: tuple[str, ...] = Field(default_factory=tuple)

    model_config = {"frozen": True, "extra": "forbid"}


class ConfidencePolicy(BaseModel):
    """Versioned v1 thresholds for the confidence ruleset (§34)."""

    policy_id: str = "llmtrace-confidence-v1"
    policy_version: str = "1.0.0"
    minimum_identities_for_high: int = Field(default=5, ge=1)
    low_execution_coverage_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    degraded_failure_ratio_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    mostly_complete_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

    model_config = {"frozen": True, "extra": "forbid"}

    @classmethod
    def create_v1(cls) -> ConfidencePolicy:
        return cls()

    def assess(
        self,
        *,
        measurement: BenchmarkMeasurementSummary | None,
        capability_profile: CapabilityProfile | None,
    ) -> ConfidenceAssessment:
        """Deterministic v1 rules:

        - Unavailable: no graded measurement at all.
        - Low: execution coverage < 50% or >= 25% provider failures, or a
          mostly-incomplete unanchored measurement.
        - Medium: the measurement is largely complete but not both fully
          healthy *and* formally calibrated against enough identities.
        - High: 32/32 graded, 0 failures, valid calibration, and
          >= ``minimum_identities_for_high`` reference identities.
        """
        reasons: list[str] = []

        if measurement is None or measurement.graded_item_count == 0:
            reasons.append("no graded benchmark measurement available")
            return ConfidenceAssessment(
                level=ConfidenceLevel.UNAVAILABLE,
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                measurement_available=False,
                calibrated=False,
                reasons=tuple(reasons),
            )

        total = measurement.total_item_count
        graded = measurement.graded_item_count
        failures = measurement.failure_item_count
        execution_coverage = measurement.execution_coverage
        graded_ratio = graded / total if total > 0 else 0.0
        failure_ratio = failures / total if total > 0 else 0.0

        calibrated = capability_profile is not None and capability_profile.calibration is not None
        identity_count = None
        if calibrated and capability_profile is not None and capability_profile.calibration is not None:
            identity_count = capability_profile.calibration.reference_identity_count

        reasons.append(f"{graded}/{total} items graded, {failures} provider failures")

        # Fail-closed directions dominate: a badly degraded measurement can
        # never be trusted enough to label Medium or High.
        if execution_coverage < self.low_execution_coverage_threshold or (
            failure_ratio >= self.degraded_failure_ratio_threshold
        ):
            reasons.append(f"measurement degraded (coverage {execution_coverage:.0%})")
            return ConfidenceAssessment(
                level=ConfidenceLevel.LOW,
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                measurement_available=True,
                calibrated=calibrated,
                reference_identity_count=identity_count,
                total_item_count=total,
                graded_item_count=graded,
                failure_item_count=failures,
                reasons=tuple(reasons),
            )

        if not calibrated:
            reasons.append("raw capability only (no reference calibration)")
            if graded_ratio >= self.mostly_complete_threshold:
                level = ConfidenceLevel.MEDIUM
            else:
                level = ConfidenceLevel.LOW
                reasons.append("mostly-incomplete measurement without a reference anchor")
        elif (
            identity_count is not None
            and identity_count >= self.minimum_identities_for_high
            and graded == total
            and failures == 0
        ):
            level = ConfidenceLevel.HIGH
            reasons.append(f"formally calibrated against {identity_count} reference identities")
        elif graded_ratio >= self.mostly_complete_threshold:
            level = ConfidenceLevel.MEDIUM
            reasons.append("reference coverage weaker than the high bar")
        else:
            level = ConfidenceLevel.LOW
            reasons.append("mostly-incomplete calibrated measurement")

        return ConfidenceAssessment(
            level=level,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            measurement_available=True,
            calibrated=calibrated,
            reference_identity_count=identity_count,
            total_item_count=total,
            graded_item_count=graded,
            failure_item_count=failures,
            reasons=tuple(reasons),
        )
