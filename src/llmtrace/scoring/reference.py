"""Reference Model Snapshot domain — immutable, append-only reference profiles.

A ``ReferenceSnapshot`` records a known reference model's ``CapabilityProfile``
at a point in time.  Reference data is historical fact: snapshots are immutable,
never overwritten, and a new measurement produces a new ``snapshot_id`` (version
append).  Storage is JSON fixture files — no database in v0.3-C.

Append-only invariant
---------------------
``same snapshot_id == same historical record``.  Overwriting is rejected
whether the previous record lives only in memory, only on disk, or in another
process.  Persistence therefore uses exclusive-create (``"x"`` mode) rather
than truncating writes, and the in-memory index is only updated *after* the
disk write succeeds.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from llmtrace.scoring.models import CapabilityProfile

from .errors import DuplicateSnapshotError, ReferenceNotFoundError

# ---------------------------------------------------------------------------
# Field validators
# ---------------------------------------------------------------------------

# snapshot_id doubles as a filename stem — it must never be able to escape or
# nest inside the repository directory.
_SNAPSHOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# A field named suite_sha256 must hold an actual SHA-256 digest.
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _require_utc(value: datetime, field_name: str) -> datetime:
    """Reject naive datetimes and normalise aware datetimes to UTC.

    Historical timestamps must carry an unambiguous instant; a naive datetime
    silently inherits the reader's local timezone and corrupts provenance.
    """
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware, got naive datetime {value.isoformat()}")
    return value.astimezone(UTC)


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
    created_at: datetime = Field(..., description="When the reference was measured (UTC, timezone-aware)")
    suite_sha256: str = Field(..., description="SHA-256 (64 hex chars) of the suite content used")
    benchmark_revision: str = Field(..., min_length=1, description="Benchmark data revision used")
    runner_version: str = Field(..., min_length=1, description="LLMTrace runner version")

    model_config = {"frozen": True, "extra": "forbid"}

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, v: datetime) -> datetime:
        """Reject naive datetimes; normalise aware datetimes to UTC."""
        return _require_utc(v, "created_at")

    @field_validator("suite_sha256")
    @classmethod
    def _validate_suite_sha256(cls, v: str) -> str:
        """Require a real 64-character hex SHA-256 digest, normalised to lowercase."""
        if not _SHA256_RE.match(v):
            raise ValueError(
                f"suite_sha256 must be exactly 64 hexadecimal characters (SHA-256), got {len(v)} chars: {v!r}"
            )
        return v.lower()


# ---------------------------------------------------------------------------
# Reference snapshot
# ---------------------------------------------------------------------------


class ReferenceSnapshot(BaseModel):
    """Immutable snapshot of a known reference model's capability profile.

    ``model_id`` is only a label (e.g. ``gpt-x``) — it is NOT an identity
    verification result.  ``capability_profile`` reuses ``CapabilityProfile``
    directly rather than duplicating it.
    """

    snapshot_id: str = Field(
        ...,
        min_length=1,
        description="Unique snapshot identifier; also a filename-safe stem",
    )
    model_id: str = Field(..., min_length=1, description="Reference model label (NOT an identity result)")
    provider_id: str = Field(..., min_length=1, description="e.g. 'openai', 'anthropic', 'moonshot'")
    created_at: datetime = Field(..., description="Snapshot creation time (UTC, timezone-aware)")
    suite_id: str = Field(..., min_length=1, description="Suite the profile was measured on")
    suite_version: str = Field(..., min_length=1, description="Suite version the profile was measured on")
    capability_profile: CapabilityProfile = Field(..., description="Reused capability profile (not copied)")
    provenance: ReferenceProvenance = Field(..., description="Auditable origin metadata")

    model_config = {"frozen": True, "extra": "forbid"}

    @field_validator("snapshot_id")
    @classmethod
    def _validate_snapshot_id(cls, v: str) -> str:
        """Require a filename-safe logical id (no separators, no traversal, no dot-only)."""
        if not _SNAPSHOT_ID_RE.match(v):
            raise ValueError(f"snapshot_id must match [A-Za-z0-9][A-Za-z0-9._-]* to stay filename-safe, got {v!r}")
        return v

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, v: datetime) -> datetime:
        """Reject naive datetimes; normalise aware datetimes to UTC."""
        return _require_utc(v, "created_at")


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class ReferenceRepository:
    """JSON-file-backed, append-only store for ``ReferenceSnapshot`` instances.

    Responsibilities are limited to ``save`` / ``get`` / ``list`` /
    ``find_by_model``.  The repository MUST NOT compute scores or perform
    comparisons — that belongs to ``CapabilityComparator``.

    Append-only contract enforced by :meth:`save`:

    1. Reject an id already present in the in-memory index.
    2. If persistent, exclusive-create the JSON file (``open("x")``) so an
       existing file — loaded or not, from any process — raises
       ``DuplicateSnapshotError`` instead of being overwritten.
    3. Only after the disk write succeeds, register the snapshot in memory.

    A failed write therefore leaves the in-memory index untouched.
    """

    def __init__(self, *, directory: Path | None = None) -> None:
        self._directory = directory
        self._snapshots: dict[str, ReferenceSnapshot] = {}

    # -- Write -------------------------------------------------------------

    def save(self, snapshot: ReferenceSnapshot) -> ReferenceSnapshot:
        """Store *snapshot*, rejecting an already-existing ``snapshot_id``.

        Snapshots are append-only; a new version must use a new ``snapshot_id``.
        The check covers the in-memory index, the on-disk file, and concurrent
        writers — whichever saw the id first.

        Raises:
            DuplicateSnapshotError: If the ``snapshot_id`` already exists in
                memory or on disk.
        """
        if snapshot.snapshot_id in self._snapshots:
            raise DuplicateSnapshotError(
                f"ReferenceSnapshot '{snapshot.snapshot_id}' already exists; snapshots are immutable and "
                f"append-only — create a new snapshot_id for a new version"
            )

        # Disk first, memory second: a failed write must not contaminate the index.
        if self._directory is not None:
            self._write_file(snapshot)

        self._snapshots[snapshot.snapshot_id] = snapshot
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
        """Resolve the JSON path for *snapshot_id*, enforcing directory containment.

        Defence in depth: even if the ``snapshot_id`` validator were ever
        relaxed, the resolved target must remain a direct child of the
        repository directory.  Traversal and nesting are both rejected.
        """
        assert self._directory is not None
        base = self._directory.resolve()
        target = (base / f"{snapshot_id}.json").resolve()
        if target.parent != base:
            raise ValueError(f"ReferenceSnapshot id '{snapshot_id}' resolves outside the repository directory '{base}'")
        return target

    def _write_file(self, snapshot: ReferenceSnapshot) -> None:
        """Exclusively create the JSON file for *snapshot*.

        ``"x"`` mode fails with ``FileExistsError`` when the file is already
        present, which is what makes the store append-only against other
        processes as well as against an empty in-memory index.
        """
        assert self._directory is not None
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._file_path(snapshot.snapshot_id)
        try:
            with path.open("x", encoding="utf-8") as f:
                f.write(snapshot.model_dump_json(indent=2))
        except FileExistsError as exc:
            raise DuplicateSnapshotError(
                f"ReferenceSnapshot '{snapshot.snapshot_id}' already exists on disk at '{path}'; "
                f"snapshots are immutable and append-only — create a new snapshot_id for a new version"
            ) from exc

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
