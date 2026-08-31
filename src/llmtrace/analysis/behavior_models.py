"""Behavior Drift domain models.

This module is the data layer for **Benchmark / Behavioral Drift** — a
concept deliberately separate from the protocol/operational drift in
``analysis/drift.py`` and from the reference-model snapshots in
``scoring/reference.py``.

A ``BehaviorRunSnapshot`` records **one observed execution** of a target API
(the Quick Suite + capability profile + HTTP evidence).  Two such snapshots
are compared by ``behavior_drift.py`` under a versioned policy.

Key invariants enforced here:

- A ``BehaviorItemKey`` is ``(task_id, source_sample_id, input_sha256)`` —
  the only stable, cross-run identity for a benchmark item.  ``attempt_id``
  is per-run and ``item_id`` is per-run ordering; neither is stable.
- ``input_sha256`` must be a real lowercase SHA-256 digest.
- Output text is never stored — only its canonicalized SHA-256 and length.
- ``created_at`` must be timezone-aware and is normalised to UTC.
- ``generation_config_sha256`` is a deterministic digest of the canonical
  generation configuration, so drift is only meaningful under identical
  generation conditions.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from llmtrace.benchmarks.models import ItemStatus, normalize_evidence_tuple
from llmtrace.scoring.models import CapabilityProfile

# ---------------------------------------------------------------------------
# Shared validators
# ---------------------------------------------------------------------------

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_sha256(value: str, field_name: str) -> str:
    """Require a real lowercase 64-hex SHA-256 digest."""
    if not _SHA256_RE.match(value):
        raise ValueError(f"{field_name} must be a lowercase 64-hex SHA-256, got {len(value)} chars: {value!r}")
    return value


def _require_utc(value: datetime, field_name: str) -> datetime:
    """Reject naive datetimes; normalise aware datetimes to UTC."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware, got naive datetime {value.isoformat()}")
    return value.astimezone(UTC)


def canonicalize_output(text: str) -> str:
    """Canonicalize raw model output for hashing.

    The minimal normalization used across drift comparisons::

        1. CRLF / CR → LF
        2. strip leading / trailing whitespace
        3. (caller UTF-8 encodes and hashes)

    It deliberately does NOT lowercase, remove internal whitespace, strip
    punctuation, or do any semantic rewriting — we only need enough
    normalization so that insignificant trailing newlines do not count as a
    behavior change.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def output_text_sha256(text: str) -> str:
    """Return the SHA-256 of the canonicalized output text."""
    return hashlib.sha256(canonicalize_output(text).encode("utf-8")).hexdigest()


def generation_config_sha256(config: object) -> str:
    """Return a deterministic SHA-256 for a generation configuration.

    Accepts either a JSON-serializable ``Mapping`` or a Pydantic model (e.g.
    ``CompletionOptions``).  Serialization is canonical: ``sort_keys=True``
    with stable separators, so dict key order does not affect the digest.
    """
    if isinstance(config, BaseModel):
        payload: object = config.model_dump(mode="json")
    elif isinstance(config, Mapping):
        payload = dict(config)
    else:
        raise ValueError(f"generation config must be a Mapping or Pydantic model, got {type(config).__name__}")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BehaviorDriftError(Exception):
    """Base exception for Behavior Drift."""


class BehaviorDriftCompatibilityError(BehaviorDriftError):
    """Base exception for two snapshots that are not comparable.

    An incomparable pair is *not* the same as an inconclusive result: it must
    fail closed by raising, never by producing a ``BehaviorDriftResult``.
    """

    error_code = "BEHAVIOR_DRIFT_INCOMPATIBLE"


class BehaviorSuiteMismatchError(BehaviorDriftCompatibilityError):
    error_code = "BEHAVIOR_SUITE_MISMATCH"


class BehaviorSuiteVersionMismatchError(BehaviorDriftCompatibilityError):
    error_code = "BEHAVIOR_SUITE_VERSION_MISMATCH"


class BehaviorSourceMismatchError(BehaviorDriftCompatibilityError):
    error_code = "BEHAVIOR_SOURCE_MISMATCH"


class BehaviorAdapterMismatchError(BehaviorDriftCompatibilityError):
    error_code = "BEHAVIOR_ADAPTER_MISMATCH"


class BehaviorScoringPolicyMismatchError(BehaviorDriftCompatibilityError):
    error_code = "BEHAVIOR_SCORING_POLICY_MISMATCH"


class GenerationConfigMismatchError(BehaviorDriftCompatibilityError):
    error_code = "GENERATION_CONFIG_MISMATCH"


class BehaviorItemSetMismatchError(BehaviorDriftCompatibilityError):
    error_code = "BEHAVIOR_ITEM_SET_MISMATCH"


class BehaviorCoverageMismatchError(BehaviorDriftCompatibilityError):
    error_code = "BEHAVIOR_COVERAGE_MISMATCH"


class BehaviorSnapshotError(BehaviorDriftError):
    """Base exception for BehaviorSnapshotBuilder failures."""


class MissingItemIdentityError(BehaviorSnapshotError):
    """An item lacks source_sample_id or input_sha256."""

    error_code = "MISSING_ITEM_IDENTITY"


class DuplicateItemKeyError(BehaviorSnapshotError):
    """Two items share the same stable ``BehaviorItemKey``."""

    error_code = "DUPLICATE_ITEM_KEY"


class ItemEvidenceError(BehaviorSnapshotError):
    """An item's evidence reference is missing, unknown, or ambiguous."""

    error_code = "ITEM_EVIDENCE_ERROR"


# ---------------------------------------------------------------------------
# Behavior Drift Level
# ---------------------------------------------------------------------------


class BehaviorDriftLevel(StrEnum):
    """Drift level for benchmark behavioral drift.

    Deliberately NOT the same enum as the legacy ``DriftLevel`` in
    ``models/report.py``, whose semantics are protocol/operational.
    """

    NO_SIGNIFICANT_DRIFT = "NO_SIGNIFICANT_DRIFT"
    OBSERVED_DRIFT = "OBSERVED_DRIFT"
    MATERIAL_DRIFT = "MATERIAL_DRIFT"
    INCONCLUSIVE = "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# BehaviorItemKey — stable cross-run item identity
# ---------------------------------------------------------------------------


class BehaviorItemKey(BaseModel):
    """Stable identity for a single benchmark item across runs.

    ``attempt_id`` changes every run and ``item_id`` is per-run ordering, so
    neither proves "this is the same, unmodified question".  The stable key is
    ``(task_id, source_sample_id, input_sha256)``.
    """

    task_id: str = Field(..., min_length=1, description="Task identifier")
    source_sample_id: str = Field(..., min_length=1, description="Upstream sample identifier")
    input_sha256: str = Field(..., description="Lowercase SHA-256 of the original input prompt")

    model_config = {"frozen": True, "extra": "forbid"}

    @field_validator("task_id", "source_sample_id")
    @classmethod
    def _strip_non_empty(cls, v: str, info: Any) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError(f"{info.field_name} must not be empty or whitespace-only")
        return stripped

    @field_validator("input_sha256")
    @classmethod
    def _validate_input_sha256(cls, v: str) -> str:
        return _validate_sha256(v, "input_sha256")

    @property
    def sort_key(self) -> tuple[str, str, str]:
        """Deterministic ordering key."""
        return (self.task_id, self.source_sample_id, self.input_sha256)

    def key_string(self) -> str:
        """Deterministic string representation for display and logging."""
        return f"{self.task_id}::{self.source_sample_id}::{self.input_sha256}"


# ---------------------------------------------------------------------------
# BehaviorItemObservation — one observed item execution
# ---------------------------------------------------------------------------


class BehaviorItemObservation(BaseModel):
    """Observed behavior for a single benchmark item in a single run.

    Only hashes, lengths, and operational metadata are stored — the full
    model output lives in the Evidence chain and is NOT copied here.
    """

    key: BehaviorItemKey = Field(..., description="Stable cross-run item identity")
    status: ItemStatus = Field(..., description="Item grading status (reuses BenchmarkItemResult.status)")
    raw_score: float = Field(..., ge=0.0, le=1.0, description="Raw score [0, 1]")
    normalized_score: float = Field(..., ge=0.0, le=1.0, description="Normalized score [0, 1]")
    output_text_sha256: str = Field(..., description="SHA-256 of canonicalized output text")
    output_length: int = Field(..., ge=0, description="Length of the raw output text (characters)")
    response_body_sha256: str = Field(default="", description="SHA-256 of the full raw response body")
    response_model: str | None = Field(default=None, description="Server-reported model identifier")
    finish_reason: str | None = Field(default=None, description="Completion finish reason")
    latency_ms: float | None = Field(default=None, ge=0.0, description="Total request latency (ms)")
    input_tokens: int | None = Field(default=None, ge=0, description="Input token count")
    output_tokens: int | None = Field(default=None, ge=0, description="Output token count")
    evidence_refs: tuple[str, ...] = Field(default_factory=tuple, description="Evidence UUIDs for this item")

    model_config = {"frozen": True, "extra": "forbid"}

    @field_validator("output_text_sha256")
    @classmethod
    def _validate_output_hash(cls, v: str) -> str:
        return _validate_sha256(v, "output_text_sha256")

    @field_validator("evidence_refs")
    @classmethod
    def _validate_evidence_refs(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        return normalize_evidence_tuple(v)


# ---------------------------------------------------------------------------
# BehaviorRunSnapshot — one observed execution
# ---------------------------------------------------------------------------

_BEHAVIOR_SNAPSHOT_VERSION = "0.1.0"


class BehaviorRunSnapshot(BaseModel):
    """Immutable snapshot of one observed benchmark execution of a target API.

    NOT a reference model profile — ``ReferenceSnapshot`` records a known
    model's historical capability fact; this records a single observed run.
    """

    snapshot_version: str = Field(
        default=_BEHAVIOR_SNAPSHOT_VERSION, min_length=1, description="Snapshot schema version"
    )
    run_id: str = Field(..., min_length=1, description="Benchmark run identifier")
    target_id: str = Field(..., min_length=1, description="Stable caller-supplied target label")
    candidate_model_id: str = Field(..., min_length=1, description="Candidate model label")
    created_at: datetime = Field(..., description="Snapshot creation time (UTC, timezone-aware)")

    suite_id: str = Field(..., min_length=1, description="Suite identifier")
    suite_version: str = Field(..., min_length=1, description="Suite version")
    source_ids: tuple[str, ...] = Field(default_factory=tuple, description="Sorted distinct source ids across the run")
    source_revisions: tuple[str, ...] = Field(
        default_factory=tuple, description="Source revisions, parallel to source_ids"
    )
    adapter_id: str = Field(..., min_length=1, description="Adapter identifier")
    adapter_version: str = Field(..., min_length=1, description="Adapter version")

    scoring_policy_id: str = Field(..., min_length=1, description="Scoring policy identifier")
    scoring_policy_version: str = Field(..., min_length=1, description="Scoring policy version")
    generation_config_sha256: str = Field(..., description="SHA-256 of the canonical generation config")

    capability_profile: CapabilityProfile = Field(..., description="Aggregated capability profile")
    items: tuple[BehaviorItemObservation, ...] = Field(
        default_factory=tuple, description="Per-item observations (sorted)"
    )
    evidence_refs: tuple[str, ...] = Field(default_factory=tuple, description="Union of all item evidence refs")

    model_config = {"frozen": True, "extra": "forbid"}

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, v: datetime) -> datetime:
        return _require_utc(v, "created_at")

    @field_validator("generation_config_sha256")
    @classmethod
    def _validate_generation_config_sha256(cls, v: str) -> str:
        return _validate_sha256(v, "generation_config_sha256")

    @field_validator("evidence_refs")
    @classmethod
    def _validate_evidence_refs(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        return normalize_evidence_tuple(v)


# ---------------------------------------------------------------------------
# Versioned drift policy
# ---------------------------------------------------------------------------


class BehaviorDriftPolicy(BaseModel):
    """Versioned policy controlling how drift is interpreted.

    Thresholds are centralized and versioned — never scattered as inline
    ``if delta > ...`` in the engine.  Changing a threshold requires a new
    policy version.  These v1 thresholds are provisional and NOT the result
    of scientific calibration.
    """

    policy_id: str = Field(..., min_length=1, description="Policy identifier")
    policy_version: str = Field(..., min_length=1, description="Policy version")
    minimum_graded_overlap_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Min graded-in-both ratio for a conclusion"
    )
    material_dimension_delta: float = Field(
        ..., gt=0.0, le=1.0, description="Absolute dimension delta threshold for MATERIAL"
    )
    material_outcome_change_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Outcome change ratio threshold for MATERIAL"
    )

    model_config = {"frozen": True, "extra": "forbid"}

    @classmethod
    def create_v1(cls) -> BehaviorDriftPolicy:
        """Return the default v0.3-D drift policy (provisional thresholds)."""
        return cls(
            policy_id="llmtrace_behavior_drift_v1",
            policy_version="0.1.0",
            minimum_graded_overlap_ratio=0.5,
            material_dimension_delta=0.2,
            material_outcome_change_ratio=0.3,
        )
