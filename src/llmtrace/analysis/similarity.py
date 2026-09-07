"""Behavior similarity v1 — experimental (§37–§40).

Ranks how close one candidate behavior profile is to trusted reference
identities using a normalized weighted L1 distance over already-verified
artifacts.  No ML model, no embedding: features come from the canonical
``BehaviorRunSnapshot`` items plus the aggregated capability profile.

Fail-closed: without comparable source artifacts (no shared dimensions, no
reference data) the result is ``unavailable`` — a name match alone never
fabricates similarity (§40).

Disclaimer: behavioral similarity only.  It is *not* model identity
verification and never routing inference.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from llmtrace.analysis.behavior_models import BehaviorRunSnapshot
from llmtrace.benchmarks.models import ItemStatus
from llmtrace.scoring.models import DimensionScoreStatus

_POLICY_ID = "llmtrace-behavior-similarity-v1"
_POLICY_VERSION = "1.0.0"
_DISCLAIMER = "Behavioral similarity only. Not model identity verification."

# Feature weights (sum <= 1.0; inactive features are dropped and the active
# weight sum re-normalizes so distance stays in [0, 1]).
_WEIGHT_DIMENSIONS = 0.6
_WEIGHT_GRADED_RATIO = 0.15
_WEIGHT_LATENCY = 0.15
_WEIGHT_OUTPUT_TOKENS = 0.1

_LATENCY_CLAMP_SECONDS = 60.0
_OUTPUT_TOKEN_CLAMP = 2048.0


class BehaviorFeatureVector(BaseModel):
    """Deterministic feature vector from one comparable behavior artifact."""

    model_id: str = Field(..., min_length=1, description="Behavior label (candidate or reference model)")
    provider_id: str = Field(..., min_length=1, description="Operational/vendor label")
    item_count: int = Field(..., ge=0)
    graded_ratio: float = Field(..., ge=0.0, le=1.0)
    dimension_scores: dict[str, float] = Field(
        default_factory=dict,
        description="dimension.value → raw_normalized_score for measured dimensions",
    )
    latency_mean_s: float | None = Field(default=None, ge=0.0)
    output_tokens_mean: float | None = Field(default=None, ge=0.0)

    model_config = {"frozen": True, "extra": "forbid"}

    @classmethod
    def from_snapshot(
        cls,
        snapshot: BehaviorRunSnapshot,
        *,
        provider_id: str,
        model_id: str | None = None,
    ) -> BehaviorFeatureVector:
        """Build a feature vector from a canonical behavior snapshot.

        Dimension scores reuse the already-aggregated profile (never re-graded
        here); item-level signals (coverage, latency, token usage) come from
        the item observations.
        """
        items = snapshot.items
        graded = sum(1 for it in items if it.status == ItemStatus.GRADED)
        graded_ratio = graded / len(items) if items else 0.0

        latencies = [it.latency_ms for it in items if it.latency_ms is not None]
        outputs = [it.output_tokens for it in items if it.output_tokens is not None]

        dimensions: dict[str, float] = {}
        for d in snapshot.capability_profile.dimensions:
            if d.status in (DimensionScoreStatus.SCORED, DimensionScoreStatus.UNCALIBRATED):
                dimensions[d.dimension.value] = d.raw_normalized_score

        return cls(
            model_id=model_id if model_id is not None else snapshot.candidate_model_id,
            provider_id=provider_id,
            item_count=len(items),
            graded_ratio=graded_ratio,
            dimension_scores=dimensions,
            latency_mean_s=(sum(latencies) / len(latencies) / 1000.0) if latencies else None,
            output_tokens_mean=(sum(outputs) / len(outputs)) if outputs else None,
        )


class BehaviorSimilarityEntry(BaseModel):
    """Similarity of one reference identity to the candidate (0–100%)."""

    model_id: str = Field(..., min_length=1)
    provider_id: str = Field(..., min_length=1)
    similarity: float = Field(..., ge=0.0, le=1.0, description="1 - normalized weighted L1 distance")
    comparable_dimensions: int = Field(..., ge=0)

    model_config = {"frozen": True, "extra": "forbid"}


class BehaviorSimilarityResult(BaseModel):
    """Ranked similarity list; empty + reason when not computable (§40)."""

    policy_id: str = Field(default=_POLICY_ID)
    policy_version: str = Field(default=_POLICY_VERSION)
    disclaimer: str = Field(default=_DISCLAIMER)
    entries: tuple[BehaviorSimilarityEntry, ...] = Field(default_factory=tuple)
    unavailable: bool = Field(default=False)
    unavailable_reason: str | None = Field(default=None)

    model_config = {"frozen": True, "extra": "forbid"}


def _normalized_feature(value: float, clamp: float) -> float:
    return max(0.0, min(value, clamp)) / clamp


def _similarity_between(candidate: BehaviorFeatureVector, reference: BehaviorFeatureVector) -> float | None:
    """Normalized weighted L1 over features present on *both* sides.

    Returns None when no comparable dimension exists (fail-closed §40).
    """
    shared_dims = sorted(set(candidate.dimension_scores) & set(reference.dimension_scores))
    if not shared_dims:
        return None

    weighted_diff = 0.0
    active_weight = 0.0

    dim_unit_weight = _WEIGHT_DIMENSIONS / len(shared_dims)
    for dim in shared_dims:
        weighted_diff += dim_unit_weight * abs(
            candidate.dimension_scores[dim] - reference.dimension_scores[dim]
        )
    active_weight += _WEIGHT_DIMENSIONS

    weighted_diff += _WEIGHT_GRADED_RATIO * abs(candidate.graded_ratio - reference.graded_ratio)
    active_weight += _WEIGHT_GRADED_RATIO

    if candidate.latency_mean_s is not None and reference.latency_mean_s is not None:
        c_lat = _normalized_feature(candidate.latency_mean_s, _LATENCY_CLAMP_SECONDS)
        r_lat = _normalized_feature(reference.latency_mean_s, _LATENCY_CLAMP_SECONDS)
        weighted_diff += _WEIGHT_LATENCY * abs(c_lat - r_lat)
        active_weight += _WEIGHT_LATENCY

    if candidate.output_tokens_mean is not None and reference.output_tokens_mean is not None:
        c_tok = _normalized_feature(candidate.output_tokens_mean, _OUTPUT_TOKEN_CLAMP)
        r_tok = _normalized_feature(reference.output_tokens_mean, _OUTPUT_TOKEN_CLAMP)
        weighted_diff += _WEIGHT_OUTPUT_TOKENS * abs(c_tok - r_tok)
        active_weight += _WEIGHT_OUTPUT_TOKENS

    if active_weight <= 0.0:
        return None
    distance = weighted_diff / active_weight
    return max(0.0, 1.0 - distance)


def assess_behavior_similarity(
    candidate: BehaviorFeatureVector,
    references: Sequence[tuple[str, str, BehaviorFeatureVector]],
) -> BehaviorSimilarityResult:
    """Compare the candidate against reference identities (model, provider, vector).

    Identities without a shared comparable dimension are skipped; if nothing
    is comparable the result is unavailable with the reason spelled out.
    """
    entries: list[BehaviorSimilarityEntry] = []
    for model_id, provider_id, reference_vector in references:
        similarity = _similarity_between(candidate, reference_vector)
        if similarity is None:
            continue
        entries.append(
            BehaviorSimilarityEntry(
                model_id=model_id,
                provider_id=provider_id,
                similarity=similarity,
                comparable_dimensions=len(
                    set(candidate.dimension_scores) & set(reference_vector.dimension_scores)
                ),
            )
        )

    entries.sort(key=lambda e: (-e.similarity, e.model_id, e.provider_id))

    if not entries:
        return BehaviorSimilarityResult(
            unavailable=True,
            unavailable_reason="no comparable reference behavior data (shared comparable dimensions required)",
        )
    return BehaviorSimilarityResult(entries=tuple(entries))
