"""ReferenceSet domain — immutable collections of verified reference snapshots.

A ``ReferenceSet`` answers "which reference facts jointly define this
Reference Universe version?"  Every member snapshot must pass the 12-gate
Compatibility check and its on-disk file must verify against its recorded
SHA-256 before it may join a trusted set.

Production builders reject ``source_type == test_fixture`` snapshots
outright (§22) — a test fixture is a test double, never a trusted reference
fact.  Tests use a dedicated helper flag that the CLI / capture service
never expose.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from llmtrace.scoring.reference import ReferenceSnapshot
from llmtrace.utilities.hashing import sha256_hash as sha256_of

from .builder import TEST_FIXTURE

# ---------------------------------------------------------------------------
# Identity / digest rules
# ---------------------------------------------------------------------------

# set_id and set_version double as filename components — never allow path
# separators, traversal, or dot-only values.
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Every *_sha256 field must hold an actual SHA-256 digest.
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _normalize_sha256(value: str, field_name: str) -> str:
    """Require a real 64-char hex SHA-256 digest, normalised to lowercase."""
    if not _SHA256_RE.match(value):
        raise ValueError(
            f"{field_name} must be exactly 64 hexadecimal characters (SHA-256), got {len(value)} chars: {value!r}"
        )
    return value.lower()


def _require_utc(value: datetime, field_name: str) -> datetime:
    """Reject naive datetimes; normalise aware datetimes to UTC."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware, got naive datetime {value.isoformat()}")
    return value.astimezone(UTC)


def _json_safe(value: Any) -> Any:
    """Recursively convert a pydantic dump into JSON-stable primitives.

    Datetimes are emitted as UTC ISO-8601 with a ``Z`` suffix so the canonical
    payload is byte-identical regardless of the original offset form.
    """
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _compute_content_sha256(payload: dict[str, Any]) -> str:
    """Canonical self-checksum (§24): ``content_sha256=""`` → stable JSON → SHA-256.

    Never hashes the Python repr — always stable JSON with sorted keys,
    compact separators, and ASCII encoding.
    """
    canonical_payload = dict(payload)
    canonical_payload["content_sha256"] = ""
    canonical = json.dumps(_json_safe(canonical_payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_of(canonical)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ReferenceSetError(Exception):
    """Base exception for ReferenceSet errors."""


class DuplicateReferenceSetError(ReferenceSetError):
    """Raised when saving a ``(reference_set_id, reference_set_version)`` that already exists.

    Reference sets are immutable and append-only — a new revision must use a
    new version string, never overwrite an existing record.
    """


class ReferenceSetNotFoundError(ReferenceSetError):
    """Raised when a requested ``(reference_set_id, reference_set_version)`` does not exist."""


class ReferenceSetIntegrityError(ReferenceSetError):
    """Raised when a set's on-disk bytes or declared content hash do not match."""


class ReferenceSetCompatibilityError(ReferenceSetError):
    """Raised when member snapshots fail the ReferenceSet Compatibility Gate."""


class FixtureReferenceError(ReferenceSetError):
    """Raised when a ``test_fixture`` snapshot is offered to a trusted ReferenceSet."""


# ---------------------------------------------------------------------------
# Member
# ---------------------------------------------------------------------------


class ReferenceSetMember(BaseModel):
    """One immutable pointer into a verified ``ReferenceSnapshot`` file.

    ``snapshot_sha256`` is the SHA-256 of the snapshot JSON *file bytes*
    (§17.1) — reading the set later must detect any manual modification of
    that file.
    """

    snapshot_id: str = Field(..., min_length=1, description="ReferenceSnapshot id this member points at")
    snapshot_sha256: str = Field(..., description="SHA-256 of the persisted snapshot file bytes")
    model_id: str = Field(..., min_length=1, description="Reference model label")
    provider_id: str = Field(..., min_length=1, description="Operational/vendor label of the reference source")
    execution_id: str | None = Field(default=None, description="RunArtifact execution id that produced the snapshot")
    capability_profile_sha256: str = Field(..., description="SHA-256 of the persisted capability_profile.json artifact")

    model_config = {"frozen": True, "extra": "forbid"}

    @field_validator("snapshot_sha256", "capability_profile_sha256")
    @classmethod
    def _validate_sha256(cls, v: str, info: ValidationInfo) -> str:
        """Require real 64-char hex digests, normalised to lowercase."""
        field_name = info.field_name
        assert field_name is not None
        return _normalize_sha256(v, field_name)


# ---------------------------------------------------------------------------
# ReferenceSet
# ---------------------------------------------------------------------------


class ReferenceSet(BaseModel):
    """Immutable set of reference snapshots defining one Reference Universe version.

    Every consistency field is copied from the *verified* member snapshots —
    never trusted from the caller — so a set is self-describing.  The
    ``content_sha256`` is a canonical self-checksum: any load that recomputes
    a different digest fails closed (§24).
    """

    reference_set_id: str = Field(..., min_length=1, description="Unique set identifier; filename-safe")
    reference_set_version: str = Field(..., min_length=1, description="Set revision; filename-safe")
    created_at: datetime = Field(..., description="Set creation time (UTC, timezone-aware)")

    suite_id: str = Field(..., min_length=1, description="Suite all members were measured on")
    suite_version: str = Field(..., min_length=1, description="Suite version all members were measured on")
    suite_content_sha256: str = Field(..., description="Suite content identity shared by all members")

    adapter_id: str = Field(..., min_length=1, description="Adapter all members ran under")
    adapter_version: str = Field(..., min_length=1, description="Adapter version all members ran under")

    scoring_policy_id: str = Field(..., min_length=1, description="Scoring policy all members used")
    scoring_policy_version: str = Field(..., min_length=1, description="Scoring policy version all members used")

    generation_config_sha256: str = Field(..., description="Generation config identity shared by all members")

    qualification_policy_id: str = Field(..., min_length=1, description="Qualification policy all members passed")
    qualification_policy_version: str = Field(..., min_length=1, description="Qualification policy version")

    members: tuple[ReferenceSetMember, ...] = Field(..., description="Verified member snapshots")

    description: str = Field(default="", description="Human-readable description")
    content_sha256: str = Field(..., description="Canonical self-checksum of the set content")

    model_config = {"frozen": True, "extra": "forbid"}

    @field_validator("reference_set_id", "reference_set_version")
    @classmethod
    def _validate_id(cls, v: str, info: ValidationInfo) -> str:
        """Require filename-safe logical ids (no separators, no traversal)."""
        if not _ID_RE.match(v):
            raise ValueError(
                f"{info.field_name} must match [A-Za-z0-9][A-Za-z0-9._-]* to stay filename-safe, got {v!r}"
            )
        return v

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, v: datetime) -> datetime:
        """Reject naive datetimes; normalise aware datetimes to UTC."""
        return _require_utc(v, "created_at")

    @field_validator("suite_content_sha256", "generation_config_sha256", "content_sha256")
    @classmethod
    def _validate_sha256_fields(cls, v: str, info: ValidationInfo) -> str:
        """Require real 64-char hex digests, normalised to lowercase."""
        field_name = info.field_name
        assert field_name is not None
        return _normalize_sha256(v, field_name)

    def compute_content_sha256(self) -> str:
        """Canonical self-hash over this set's content with ``content_sha256=""`` (§24)."""
        return _compute_content_sha256(self.model_dump(exclude={"content_sha256"}))

    def verify_content_hash(self) -> str:
        """Fail closed if the declared ``content_sha256`` does not recompute.

        Returns the verified digest on success.
        """
        actual = self.compute_content_sha256()
        if actual != self.content_sha256:
            raise ReferenceSetIntegrityError(
                f"ReferenceSet '{self.reference_set_id}' v{self.reference_set_version} content hash mismatch: "
                f"recomputed {actual!r} != declared {self.content_sha256!r}"
            )
        return actual


# ---------------------------------------------------------------------------
# Builder (Compatibility Gate)
# ---------------------------------------------------------------------------

_COMPARABLE_STATUSES = frozenset({"scored", "uncalibrated"})


def _comparable_dimensions(snapshot: ReferenceSnapshot) -> frozenset[str]:
    """Dimensions that carry a real measurement (SCORED / UNCALIBRATED only)."""
    return frozenset(
        d.dimension.value for d in snapshot.capability_profile.dimensions if d.status.value in _COMPARABLE_STATUSES
    )


class ReferenceSetBuilder:
    """Compatibility-gated builder for trusted ReferenceSets (§21–§24).

    The builder only assembles already-verified facts: callers verify each
    snapshot's on-disk SHA against the member's recorded hash *before* calling
    :meth:`build`.  The 12-gate Compatibility check is fail-closed.
    """

    def __init__(self, *, allow_test_fixture: bool = False) -> None:
        self._allow_test_fixture = allow_test_fixture

    def build(
        self,
        *,
        reference_set_id: str,
        reference_set_version: str,
        created_at: datetime,
        snapshots: Sequence[ReferenceSnapshot],
        snapshot_sha256s: Mapping[str, str],
        description: str = "",
    ) -> ReferenceSet:
        """Assemble a ``ReferenceSet`` from verified member snapshots.

        Args:
            reference_set_id: Unique, filename-safe set identifier.
            reference_set_version: Revision; filename-safe.
            created_at: Set creation time (UTC, timezone-aware).
            snapshots: Member snapshots (same suite/policy/adapter family).
            snapshot_sha256s: ``{snapshot_id: sha256_of_persisted_file_bytes}`` —
                the caller's verification result (must cover every snapshot).
            description: Optional human-readable description.

        Raises:
            FixtureReferenceError: If a member is ``test_fixture`` and
                ``allow_test_fixture`` is False.
            ReferenceSetCompatibilityError: If any member fails the gate.
            ReferenceSetError: On empty sets, duplicate ids, or missing hashes.
        """
        member_list = list(snapshots)
        if not member_list:
            raise ReferenceSetError("cannot build a ReferenceSet with no member snapshots")

        for snapshot in member_list:
            if snapshot.provenance.source_type == TEST_FIXTURE and not self._allow_test_fixture:
                raise FixtureReferenceError(
                    f"ReferenceSnapshot '{snapshot.snapshot_id}' is a test_fixture; "
                    f"test fixtures are never trusted reference facts (§22)"
                )

        ids = [s.snapshot_id for s in member_list]
        if len(set(ids)) != len(ids):
            raise ReferenceSetError(f"duplicate snapshot_id in ReferenceSet members: {ids}")
        missing = [sid for sid in ids if sid not in snapshot_sha256s]
        if missing:
            raise ReferenceSetError(f"missing verified snapshot SHA-256 for member(s): {missing}")

        # Deterministic member order so the content hash is stable.
        ordered = sorted(member_list, key=lambda s: s.snapshot_id)
        base = ordered[0]
        for snapshot in ordered[1:]:
            self._assert_compatible(base, snapshot)

        members = tuple(self._member_for(snapshot, snapshot_sha256s[snapshot.snapshot_id]) for snapshot in ordered)

        payload: dict[str, Any] = {
            "reference_set_id": reference_set_id,
            "reference_set_version": reference_set_version,
            "created_at": created_at,
            "suite_id": base.suite_id,
            "suite_version": base.suite_version,
            "suite_content_sha256": base.provenance.suite_sha256,
            "adapter_id": self._require(base, "adapter_id"),
            "adapter_version": self._require(base, "adapter_version"),
            "scoring_policy_id": base.capability_profile.scoring_policy_id,
            "scoring_policy_version": base.capability_profile.scoring_policy_version,
            "generation_config_sha256": self._require(base, "generation_config_sha256"),
            "qualification_policy_id": self._require(base, "qualification_policy_id"),
            "qualification_policy_version": self._require(base, "qualification_policy_version"),
            "members": [m.model_dump() for m in members],
            "description": description,
        }
        content_sha256 = _compute_content_sha256(payload)
        return ReferenceSet(**payload, content_sha256=content_sha256)

    # -- Internals ---------------------------------------------------------

    def _require(self, snapshot: ReferenceSnapshot, field: str) -> str:
        value = getattr(snapshot.provenance, field)
        if value is None:
            raise ReferenceSetCompatibilityError(
                f"ReferenceSnapshot '{snapshot.snapshot_id}' provenance.{field} is missing; "
                f"a trusted ReferenceSet requires it"
            )
        return str(value)

    def _member_for(self, snapshot: ReferenceSnapshot, snapshot_sha256: str) -> ReferenceSetMember:
        profile_sha = snapshot.provenance.capability_profile_sha256
        if profile_sha is None:
            raise ReferenceSetCompatibilityError(
                f"ReferenceSnapshot '{snapshot.snapshot_id}' has no capability_profile_sha256; "
                f"a trusted ReferenceSet member requires it"
            )
        return ReferenceSetMember(
            snapshot_id=snapshot.snapshot_id,
            snapshot_sha256=snapshot_sha256,
            model_id=snapshot.model_id,
            provider_id=snapshot.provider_id,
            execution_id=snapshot.provenance.execution_id,
            capability_profile_sha256=profile_sha,
        )

    def _assert_compatible(self, base: ReferenceSnapshot, other: ReferenceSnapshot) -> None:
        """Fail closed on any of the 12 compatibility gates (§21)."""
        mismatches: list[str] = []

        def check(name: str, actual: object, expected: object) -> None:
            if actual != expected:
                mismatches.append(f"{name}: {actual!r} != {expected!r}")

        check("suite_id", other.suite_id, base.suite_id)
        check("suite_version", other.suite_version, base.suite_version)
        check("suite_content_sha256", other.provenance.suite_sha256, base.provenance.suite_sha256)
        check("adapter_id", other.provenance.adapter_id, base.provenance.adapter_id)
        check("adapter_version", other.provenance.adapter_version, base.provenance.adapter_version)
        check(
            "scoring_policy_id",
            other.capability_profile.scoring_policy_id,
            base.capability_profile.scoring_policy_id,
        )
        check(
            "scoring_policy_version",
            other.capability_profile.scoring_policy_version,
            base.capability_profile.scoring_policy_version,
        )
        check(
            "generation_config_sha256",
            other.provenance.generation_config_sha256,
            base.provenance.generation_config_sha256,
        )
        check(
            "qualification_policy_id",
            other.provenance.qualification_policy_id,
            base.provenance.qualification_policy_id,
        )
        check(
            "qualification_policy_version",
            other.provenance.qualification_policy_version,
            base.provenance.qualification_policy_version,
        )

        dims_actual = _comparable_dimensions(other)
        dims_base = _comparable_dimensions(base)
        if dims_actual != dims_base:
            mismatches.append(f"comparable_dimensions: {sorted(dims_actual)} != {sorted(dims_base)}")
        if not math.isclose(
            other.capability_profile.coverage_weight,
            base.capability_profile.coverage_weight,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            mismatches.append(
                f"coverage_weight: {other.capability_profile.coverage_weight} != "
                f"{base.capability_profile.coverage_weight}"
            )

        if mismatches:
            raise ReferenceSetCompatibilityError(
                f"ReferenceSnapshot '{other.snapshot_id}' is incompatible with '{base.snapshot_id}': "
                + "; ".join(mismatches)
            )


__all__: list[str] = [
    "ReferenceSetError",
    "DuplicateReferenceSetError",
    "ReferenceSetNotFoundError",
    "ReferenceSetIntegrityError",
    "ReferenceSetCompatibilityError",
    "FixtureReferenceError",
    "ReferenceSetMember",
    "ReferenceSet",
    "ReferenceSetBuilder",
]
