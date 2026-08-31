"""Central evidence recording — Provider-level exactly-once collection.

Quick Suite (and any future adapter) receives only evidence *ids* back from
``Provider.complete()``; the real ``HTTPEvidence`` objects would otherwise be
lost to the orchestration layer.  The recorder closes that gap: the provider
records every real HTTP request exactly once, in order, so the unified report
and the ``BehaviorSnapshotBuilder`` share one evidence source of truth.
"""

from __future__ import annotations

from typing import Protocol

from llmtrace.analysis.behavior_models import DuplicateEvidenceError
from llmtrace.models.evidence import HTTPEvidence


class EvidenceRecorder(Protocol):
    """Minimal central evidence collection contract."""

    def record(self, evidence: HTTPEvidence) -> None:
        """Record *evidence* exactly once, preserving arrival order."""
        ...


class InMemoryEvidenceRecorder:
    """Ordered, duplicate-rejecting in-memory evidence store.

    - Records evidence in arrival order (never reorders).
    - A duplicate ``evidence_id`` fails closed instead of silently
      last-write-wins.
    - Never mutates the recorded evidence.
    """

    def __init__(self) -> None:
        self._evidence: list[HTTPEvidence] = []
        self._seen: set[str] = set()

    def record(self, evidence: HTTPEvidence) -> None:
        evidence_id = str(evidence.evidence_id)
        if evidence_id in self._seen:
            raise DuplicateEvidenceError(
                f"duplicate evidence_id '{evidence_id}'; each real HTTP request must be recorded exactly once"
            )
        self._seen.add(evidence_id)
        self._evidence.append(evidence)

    def list(self) -> tuple[HTTPEvidence, ...]:
        """Return the recorded evidence in arrival order."""
        return tuple(self._evidence)

    def snapshot(self) -> tuple[HTTPEvidence, ...]:
        """Alias of :meth:`list` for explicit copy semantics."""
        return self.list()

    def __len__(self) -> int:
        return len(self._evidence)

    def __contains__(self, evidence_id: str) -> bool:
        return evidence_id in self._seen
