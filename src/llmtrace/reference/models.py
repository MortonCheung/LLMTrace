"""Reference qualification domain models (v0.4-A).

``ReferenceQualificationResult`` is the machine-readable outcome of running
the qualification gate chain over one persisted Run Artifact.  It either
carries the verified, persisted capability profile (QUALIFIED) or a set of
machine-readable ``reason_codes`` explaining why the run may never become a
Trusted Reference (REJECTED).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from llmtrace.scoring.models import CapabilityProfile


class ReferenceQualificationStatus(StrEnum):
    """Outcome of the reference qualification gate chain."""

    QUALIFIED = "QUALIFIED"
    REJECTED = "REJECTED"


class ReferenceQualificationResult(BaseModel):
    """Result of qualifying one run artifact for reference candidacy.

    ``capability_profile`` is populated only on QUALIFIED — and it is the
    profile read from the *persisted and verified* ``capability_profile.json``
    artifact, never a transient in-memory object.
    """

    status: ReferenceQualificationStatus = Field(...)
    policy_id: str = Field(..., min_length=1, description="Qualification policy id used")
    policy_version: str = Field(..., min_length=1, description="Qualification policy version used")
    execution_id: str = Field(..., min_length=1, description="The run artifact execution that was qualified")
    reason_codes: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Machine-readable reasons; empty on QUALIFIED",
    )
    warnings: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Non-fatal warnings surfaced during qualification",
    )
    capability_profile: CapabilityProfile | None = Field(
        default=None,
        description="Persisted + verified profile; set only on QUALIFIED",
    )

    model_config = {"frozen": True, "extra": "forbid"}

    @property
    def qualified(self) -> bool:
        """True when the run passed every qualification gate."""
        return self.status == ReferenceQualificationStatus.QUALIFIED
