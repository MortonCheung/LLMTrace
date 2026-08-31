"""Reference Model Snapshot domain — immutable, append-only reference profiles.

A ``ReferenceSnapshot`` records a known reference model's ``CapabilityProfile``
at a point in time.  Reference data is historical fact: snapshots are immutable,
never overwritten, and a new measurement produces a new ``snapshot_id`` (version
append).  Storage is JSON fixture files — no database in v0.3-C.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from llmtrace.scoring.models import CapabilityProfile

from .errors import DuplicateSnapshotError, ReferenceNotFoundError

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class ReferenceProvenance(BaseModel):
    """Auditable origin metadata for a reference snapshot.

    Records who produced the reference, from what source, with what runner,
    and the exact suite content hash — so that future questions like
    "when was this reference measured, and against which suite content?"
    can be answered.  Unauditable labels (e.g. ``{"model": "GPT"}``) are
    deliberately not part of this schema.
    """

    source_type: str = Field(..., min_length=1, description="e.g. 'benchmark_run', 'manual', 'imported'")
    created_by: str = Field(..., min_length=1, description="Producer of this reference (tool or operator)")
    created_at: datetime = Field(..., description="When the reference was measured (UTC)")
    suite_sha256: str = Field(..., min_length=1, description="SHA-256 of the suite content used")
    benchmark_revision: str = Field(..., min_length=1, description="Benchmark data revision used")
    runner_version: str = Field(..., min_length=1, description="LLMTrace runner version")

    model_config = {"frozen": True, "extra": "forbid"}


# ---------------------------------------------------------------------------
# Reference snapshot
# ---------------------------------------------------------------------------


class ReferenceSnapshot(BaseModel):
    """Immutable snapshot of a known reference model's capability profile.

    ``model_id`` is only a label (e.g. ``gpt-x``) — it is NOT an identity
    verification result.  ``capability_profile`` reuses ``CapabilityProfile``
    directly rather than duplicating it.
    """

    snapshot_id: str = Field(..., min_length=1, description="Unique snapshot identifier")
    model_id: str = Field(..., min_length=1, description="Reference model label (NOT an identity result)")
    provider_id: str = Field(..., min_length=1, description="e.g. 'openai', 'anthropic', 'moonshot'")
    created_at: datetime = Field(..., description="Snapshot creation time (UTC)")
    suite_id: str = Field(..., min_length=1, description="Suite the profile was measured on")
    suite_version: str = Field(..., min_length=1, description="Suite version the profile was measured on")
    capability_profile: CapabilityProfile = Field(..., description="Reused capability profile (not copied)")
    provenance: ReferenceProvenance = Field(..., description="Auditable origin metadata")

    model_config = {"frozen": True, "extra": "forbid"}


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class ReferenceRepository:
    """JSON-file-backed store for ``ReferenceSnapshot`` instances.

    Responsibilities are limited to ``save`` / ``get`` / ``list`` /
    ``find_by_model``.  The repository MUST NOT compute scores or perform
    comparisons — that belongs to ``CapabilityComparator``.
    """

    def __init__(self, *, directory: Path | None = None) -> None:
        self._directory = directory
        self._snapshots: dict[str, ReferenceSnapshot] = {}

    # -- Write -------------------------------------------------------------

    def save(self, snapshot: ReferenceSnapshot) -> ReferenceSnapshot:
        """Store *snapshot*, rejecting an already-existing ``snapshot_id``.

        Snapshots are append-only; a new version must use a new ``snapshot_id``.
        """
        if snapshot.snapshot_id in self._snapshots:
            raise DuplicateSnapshotError(
                f"ReferenceSnapshot '{snapshot.snapshot_id}' already exists; snapshots are immutable and "
                f"append-only — create a new snapshot_id for a new version"
            )
        self._snapshots[snapshot.snapshot_id] = snapshot
        if self._directory is not None:
            self._write_file(snapshot)
        return snapshot

    # -- Read --------------------------------------------------------------

    def get(self, snapshot_id: str) -> ReferenceSnapshot:
        """Return the snapshot with *snapshot_id*.

        Raises:
            ReferenceNotFoundError: If no snapshot with that id exists.
        """
        try:
            return self._snapshots[snapshot_id]
        except KeyError as exc:
            raise ReferenceNotFoundError(f"ReferenceSnapshot '{snapshot_id}' not found") from exc

    def list(self) -> Sequence[ReferenceSnapshot]:
        """Return all snapshots sorted by ``snapshot_id``."""
        return sorted(self._snapshots.values(), key=lambda s: s.snapshot_id)

    def find_by_model(self, model_id: str) -> Sequence[ReferenceSnapshot]:
        """Return all snapshots whose ``model_id`` label matches *model_id*."""
        return [s for s in self.list() if s.model_id == model_id]

    def __len__(self) -> int:
        return len(self._snapshots)

    def __contains__(self, snapshot_id: str) -> bool:
        return snapshot_id in self._snapshots

    @classmethod
    def load(cls, directory: Path) -> ReferenceRepository:
        """Build a repository by loading every ``*.json`` fixture in *directory*."""
        repo = cls(directory=directory)
        repo._load_directory(directory)
        return repo

    # -- File I/O ----------------------------------------------------------

    def _file_path(self, snapshot_id: str) -> Path:
        assert self._directory is not None
        return self._directory / f"{snapshot_id}.json"

    def _write_file(self, snapshot: ReferenceSnapshot) -> None:
        assert self._directory is not None
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._file_path(snapshot.snapshot_id)
        path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")

    def _load_directory(self, directory: Path) -> None:
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            snapshot = ReferenceSnapshot.model_validate(data)
            if snapshot.snapshot_id in self._snapshots:
                raise DuplicateSnapshotError(
                    f"Duplicate ReferenceSnapshot '{snapshot.snapshot_id}' found in '{directory}'"
                )
            self._snapshots[snapshot.snapshot_id] = snapshot
