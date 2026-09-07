"""Routing stability v1 — experimental (§41–§42).

First version does *not* try to infer a routing mix (e.g. "30% GPT-X").  It
only labels how consistent the served behavior looks over one run:

    Stable / Mostly Stable / Suspicious / Insufficient Data

Signals come from already-recorded observations: server-reported model
identifier variation, provider failures, latency clusters and token spread
(both currently reported, not used to *decide* yet — the v1 decision is
documented below), behavior drift and repeat samples stay out of scope for
v1 and land in later phases.

Text discipline: never a fabricated mixing percentage.  Suspicious wording is
limited to "request behavior is inconsistent across samples; mixed routing is
possible" (§42).
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum

from pydantic import BaseModel, Field

from llmtrace.analysis.behavior_models import BehaviorRunSnapshot
from llmtrace.benchmarks.models import ItemStatus

_POLICY_ID = "llmtrace-routing-stability-v1"
_POLICY_VERSION = "1.0.0"
_MINIMUM_SAMPLES = 8
_MINIMUM_DOMINANT_RATIO = 0.9
_SUSPICIOUS_FAILURE_RATIO = 0.25


class RoutingStabilityLevel(StrEnum):
    STABLE = "Stable"
    MOSTLY_STABLE = "Mostly Stable"
    SUSPICIOUS = "Suspicious"
    INSUFFICIENT_DATA = "Insufficient Data"


class RoutingAssessment(BaseModel):
    """One routing-stability verdict for a single run's behavior."""

    level: RoutingStabilityLevel = Field(...)
    policy_id: str = Field(default=_POLICY_ID)
    policy_version: str = Field(default=_POLICY_VERSION)
    experimental: bool = Field(default=True, description="Experimental label, not statistical proof")
    item_count: int = Field(..., ge=0)
    distinct_response_models: int = Field(..., ge=0)
    dominant_model_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    failure_ratio: float = Field(..., ge=0.0, le=1.0)
    reasons: tuple[str, ...] = Field(default_factory=tuple)

    model_config = {"frozen": True, "extra": "forbid"}


def assess_routing_stability(snapshot: BehaviorRunSnapshot) -> RoutingAssessment:
    """Deterministic v1 rules:

    - Insufficient Data: fewer than 8 items, or the server never reported a
      model identifier on any item (fail-closed, §42).
    - Stable: a single reported model and zero provider failures.
    - Mostly Stable: one reported model with a small failure ratio (< 25%),
      or one strongly dominant model (>= 90%) alongside a stray identity.
    - Suspicious: multiple models with no clear dominant one, or a single
      reported model with a high failure ratio (>= 25%) — possible failover.
    """
    items = snapshot.items
    total = len(items)
    reasons: list[str] = []

    if total < _MINIMUM_SAMPLES:
        return _verdict(RoutingStabilityLevel.INSUFFICIENT_DATA, total, reasons, message="insufficient samples")
    reasons.append(f"{total} behavior samples")

    reported = [it.response_model for it in items if it.response_model is not None]
    if not reported:
        return _verdict(
            RoutingStabilityLevel.INSUFFICIENT_DATA,
            total,
            reasons,
            message="server reported no model identifier on any item",
        )

    counts = Counter(reported)
    dominant_ratio = counts.most_common(1)[0][1] / len(reported)
    distinct = len(counts)
    reasons.append(f"{distinct} distinct reported model identifier(s)")

    failures = sum(1 for it in items if it.status == ItemStatus.FAILURE)
    failure_ratio = failures / total

    if distinct == 1 and failure_ratio == 0.0:
        level = RoutingStabilityLevel.STABLE
        reasons.append("single reported model, zero provider failures")
    elif distinct == 1:
        level = (
            RoutingStabilityLevel.MOSTLY_STABLE
            if failure_ratio < _SUSPICIOUS_FAILURE_RATIO
            else RoutingStabilityLevel.SUSPICIOUS
        )
        reasons.append(
            f"{failures}/{total} provider failures — possible failover"
            if failure_ratio >= _SUSPICIOUS_FAILURE_RATIO
            else f"{failures}/{total} provider failures, model identity stable"
        )
    elif dominant_ratio >= _MINIMUM_DOMINANT_RATIO:
        level = RoutingStabilityLevel.MOSTLY_STABLE
        reasons.append(f"one dominant model ({dominant_ratio:.0%}) with stray identifiers")
    else:
        level = RoutingStabilityLevel.SUSPICIOUS
        reasons.append("request behavior is inconsistent across samples; mixed routing is possible")

    return RoutingAssessment(
        level=level,
        item_count=total,
        distinct_response_models=distinct,
        dominant_model_ratio=dominant_ratio,
        failure_ratio=failure_ratio,
        reasons=tuple(reasons),
    )


def _verdict(
    level: RoutingStabilityLevel,
    total: int,
    reasons: list[str],
    *,
    message: str,
) -> RoutingAssessment:
    return RoutingAssessment(
        level=level,
        item_count=total,
        distinct_response_models=0,
        dominant_model_ratio=None,
        failure_ratio=0.0,
        reasons=(*reasons, message),
    )
