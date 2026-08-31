"""Append-only JSON-file-backed store for ``ReferenceSet`` instances (§25–§26).

Storage layout under the reference directory::

    references/
    ├── snapshots/          # ReferenceSnapshot files (existing ReferenceRepository)
    └── sets/               # ReferenceSet files (this module)

The repository keeps the same append-only discipline as the snapshot store:

1. Reject a ``(reference_set_id, reference_set_version)`` already in memory.
2. If persistent, exclusive-create the JSON file (``"x"`` mode) so an
   existing file — loaded or not, from any process — raises
   ``DuplicateReferenceSetError`` instead of being overwritten.
3. Only after the disk write succeeds, register the set in memory.

A failed write therefore leaves the in-memory index untouched.  The
repository is a dumb store: it never computes compatibility or hashes —
that belongs to ``ReferenceSetBuilder`` and the models.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from llmtrace.utilities.hashing import sha256_hash as sha256_of

from .reference_set import (
    DuplicateReferenceSetError,
    ReferenceSet,
    ReferenceSetIntegrityError,
    ReferenceSetNotFoundError,
)


class ReferenceSetRepository:
    """Append-only store for ``ReferenceSet`` instances.

    ``save`` / ``get`` / ``list`` / ``verify`` mirror the snapshot
    repository; ``verify`` additionally recomputes the set's canonical
    content hash so a tampered set file fails closed on read.
    """

    def __init__(self, *, directory: Path | None = None) -> None:
        self._directory = directory
        self._sets: dict[tuple[str, str], ReferenceSet] = {}

    # -- Write -------------------------------------------------------------

    def save(self, reference_set: ReferenceSet) -> ReferenceSet:
        """Store *reference_set*, rejecting an already-existing id/version pair.

        Sets are append-only; a new revision must use a new
        ``reference_set_version``.  The check covers the in-memory index, the
        on-disk file, and concurrent writers — whichever saw the key first.

        Raises:
            DuplicateReferenceSetError: If the id/version pair already exists
                in memory or on disk.
        """
        key = (reference_set.reference_set_id, reference_set.reference_set_version)
        if key in self._sets:
            raise DuplicateReferenceSetError(
                f"ReferenceSet '{key[0]}' v{key[1]} already exists; sets are immutable and append-only — "
                f"use a new reference_set_version for a new revision"
            )

        # Disk first, memory second: a failed write must not contaminate the index.
        if self._directory is not None:
            self._write_file(reference_set)

        self._sets[key] = reference_set
        return reference_set

    # -- Read --------------------------------------------------------------

    def get(self, reference_set_id: str, reference_set_version: str) -> ReferenceSet:
        """Return the set with *reference_set_id* at *reference_set_version*.

        Raises:
            ReferenceSetNotFoundError: If no such set exists.
        """
        try:
            return self._sets[(reference_set_id, reference_set_version)]
        except KeyError as exc:
            raise ReferenceSetNotFoundError(
                f"ReferenceSet '{reference_set_id}' v{reference_set_version} not found"
            ) from exc

    def read_raw(self, reference_set_id: str, reference_set_version: str) -> str:
        """Return the raw JSON bytes of the persisted set file as text.

        Read-only integrity API (§18 discipline): never writes to disk.
        """
        if self._directory is None:
            raise ReferenceSetNotFoundError(
                f"ReferenceSet '{reference_set_id}' v{reference_set_version} has no persisted file "
                f"(in-memory repository)"
            )
        path = self._file_path(reference_set_id, reference_set_version)
        if not path.exists():
            raise ReferenceSetNotFoundError(
                f"ReferenceSet '{reference_set_id}' v{reference_set_version} not found on disk at '{path}'"
            )
        return path.read_text(encoding="utf-8")

    def set_sha256(self, reference_set_id: str, reference_set_version: str) -> str:
        """Return SHA-256 of the *persisted* set file bytes."""
        return sha256_of(self.read_raw(reference_set_id, reference_set_version))

    def verify(
        self,
        reference_set_id: str,
        reference_set_version: str,
        expected_sha256: str | None = None,
    ) -> str:
        """Verify the set's integrity, failing closed on any mismatch.

        1. Recomputed content hash must equal the declared ``content_sha256``.
        2. When persisted, the on-disk file bytes must match the recorded
           digest (*expected_sha256*, or the current in-memory serialisation).

        Returns the on-disk file SHA when persisted, otherwise the verified
        content hash.

        Raises:
            ReferenceSetIntegrityError: On any hash mismatch.
        """
        reference_set = self.get(reference_set_id, reference_set_version)
        reference_set.verify_content_hash()

        if self._directory is not None:
            actual_disk = self.set_sha256(reference_set_id, reference_set_version)
            expected = expected_sha256 or sha256_of(reference_set.model_dump_json(indent=2))
            if actual_disk != expected:
                raise ReferenceSetIntegrityError(
                    f"ReferenceSet '{reference_set_id}' v{reference_set_version} on-disk integrity check failed: "
                    f"file SHA-256 {actual_disk!r} != expected {expected!r}"
                )
            return actual_disk
        return reference_set.content_sha256

    def list(self) -> Sequence[ReferenceSet]:
        """Return all sets sorted by ``(reference_set_id, reference_set_version)``."""
        return sorted(self._sets.values(), key=lambda s: (s.reference_set_id, s.reference_set_version))

    def __len__(self) -> int:
        return len(self._sets)

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self._sets

    @classmethod
    def load(cls, directory: Path) -> ReferenceSetRepository:
        """Build a repository by loading every ``*.json`` set file in *directory*."""
        repo = cls(directory=directory)
        repo._load_directory(directory)
        return repo

    # -- File I/O ----------------------------------------------------------

    def _file_path(self, reference_set_id: str, reference_set_version: str) -> Path:
        """Resolve the JSON path for a set, enforcing directory containment.

        Defence in depth: even if the id/version validators were ever
        relaxed, the resolved target must remain a direct child of the
        repository directory.
        """
        assert self._directory is not None
        base = self._directory.resolve()
        target = (base / f"{reference_set_id}_{reference_set_version}.json").resolve()
        if target.parent != base:
            raise ValueError(
                f"ReferenceSet id '{reference_set_id}' / version '{reference_set_version}' "
                f"resolves outside the repository directory '{base}'"
            )
        return target

    def _write_file(self, reference_set: ReferenceSet) -> None:
        """Exclusively create the JSON file for *reference_set*.

        ``"x"`` mode fails with ``FileExistsError`` when the file already
        exists, which is what makes the store append-only against other
        processes as well as against an empty in-memory index.
        """
        assert self._directory is not None
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._file_path(reference_set.reference_set_id, reference_set.reference_set_version)
        try:
            with path.open("x", encoding="utf-8") as f:
                f.write(reference_set.model_dump_json(indent=2))
        except FileExistsError as exc:
            raise DuplicateReferenceSetError(
                f"ReferenceSet '{reference_set.reference_set_id}' v{reference_set.reference_set_version} "
                f"already exists on disk at '{path}'; sets are immutable and append-only — "
                f"use a new reference_set_version for a new revision"
            ) from exc

    def _load_directory(self, directory: Path) -> None:
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            reference_set = ReferenceSet.model_validate(data)
            key = (reference_set.reference_set_id, reference_set.reference_set_version)
            if key in self._sets:
                raise DuplicateReferenceSetError(f"Duplicate ReferenceSet '{key[0]}' v{key[1]} found in '{directory}'")
            self._sets[key] = reference_set


__all__: list[str] = ["ReferenceSetRepository"]
