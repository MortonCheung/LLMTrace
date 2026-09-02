"""Append-only local run artifact repository.

One execution = one immutable directory::

    reports/runs/<execution_id>/
        manifest.json          (written last)
        report.json
        report.html
        capability_profile.json
        behavior_snapshot.json
        benchmark_runs.json

Commit is staging + hash + atomic rename, mirroring the append-only
discipline of ``ReferenceRepository``: an existing execution directory is
never overwritten, and a failed commit leaves no half-written run behind.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from llmtrace.analysis.behavior_models import BehaviorRunSnapshot
from llmtrace.execution.models import MANIFEST_VERSION, RunArtifactManifest

if TYPE_CHECKING:
    from llmtrace.scoring.models import CapabilityProfile

_MANIFEST_FILENAME = "manifest.json"
# execution_id is generated internally (UUID); user input must never reach a path.
_EXECUTION_ID_RE = re.compile(r"^[0-9a-fA-F][0-9a-fA-F-]{7,63}$")
_ARTIFACT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ArtifactRepositoryError(Exception):
    """Base exception for the run artifact repository."""


class DuplicateExecutionError(ArtifactRepositoryError):
    """Raised when committing an execution_id that already exists."""

    error_code = "DUPLICATE_EXECUTION"


class ArtifactNotFoundError(ArtifactRepositoryError):
    """Raised when a requested run or artifact does not exist."""

    error_code = "ARTIFACT_NOT_FOUND"


class ArtifactIntegrityError(ArtifactRepositoryError):
    """Raised when an artifact's content no longer matches its recorded hash."""

    error_code = "ARTIFACT_INTEGRITY_FAILURE"


def sha256_of(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class RunArtifactRepository:
    """Filesystem-backed, append-only store of unified run artifacts."""

    def __init__(self, root: Path) -> None:
        self._root = root / "runs"

    # -- Paths -------------------------------------------------------------

    def _run_dir(self, execution_id: str) -> Path:
        if not _EXECUTION_ID_RE.match(execution_id):
            raise ArtifactRepositoryError(f"execution_id must be a safe generated id (UUID-like), got {execution_id!r}")
        base = self._root.resolve()
        target = (base / execution_id).resolve()
        if target.parent != base:
            raise ArtifactRepositoryError(
                f"execution_id '{execution_id}' resolves outside the repository directory '{base}'"
            )
        return target

    # -- Write -------------------------------------------------------------

    def commit(
        self,
        manifest: RunArtifactManifest,
        artifacts: dict[str, str],
    ) -> RunArtifactManifest:
        """Atomically commit one execution's artifacts; returns the final manifest.

        Raises:
            DuplicateExecutionError: If the execution directory already exists.
            ArtifactRepositoryError: On invalid names or write failures.
        """
        final_dir = self._run_dir(manifest.execution_id)
        if final_dir.exists():
            raise DuplicateExecutionError(
                f"execution '{manifest.execution_id}' already exists; run artifacts are append-only"
            )

        if not artifacts:
            raise ArtifactRepositoryError("refusing to commit a run without artifacts")

        for name in artifacts:
            if not _ARTIFACT_NAME_RE.match(name) or name == _MANIFEST_FILENAME:
                raise ArtifactRepositoryError(f"invalid artifact filename: {name!r}")

        self._root.mkdir(parents=True, exist_ok=True)
        staging = self._root / f".staging-{manifest.execution_id}"
        if staging.exists():
            shutil.rmtree(staging)

        try:
            staging.mkdir(parents=True)

            hashes: dict[str, str] = {}
            for name, content in artifacts.items():
                path = staging / name
                path.write_text(content, encoding="utf-8")
                hashes[name] = sha256_of(content)

            # Manifest is written LAST, after every artifact exists on disk.
            final_manifest = manifest.model_copy(update={"artifacts": hashes, "manifest_version": MANIFEST_VERSION})
            (staging / _MANIFEST_FILENAME).write_text(final_manifest.model_dump_json(indent=2), encoding="utf-8")

            # Atomic appearance: the run directory only ever shows up complete.
            staging.rename(final_dir)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        return final_manifest

    # -- Read --------------------------------------------------------------

    def load_manifest(self, execution_id: str) -> RunArtifactManifest:
        path = self._run_dir(execution_id) / _MANIFEST_FILENAME
        if not path.exists():
            raise ArtifactNotFoundError(f"no manifest for execution '{execution_id}'")
        return RunArtifactManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def read_manifest_raw(self, execution_id: str) -> str:
        """Return the raw ``manifest.json`` bytes as text.

        Read-only provenance API: the reference layer needs the SHA-256 of the
        *actual* manifest file bytes (not a re-serialization) to record honest
        ``run_manifest_sha256`` provenance.
        """
        path = self._run_dir(execution_id) / _MANIFEST_FILENAME
        if not path.exists():
            raise ArtifactNotFoundError(f"no manifest for execution '{execution_id}'")
        return path.read_text(encoding="utf-8")

    def manifest_sha256(self, execution_id: str) -> str:
        """SHA-256 of the actual ``manifest.json`` bytes on disk."""
        return sha256_of(self.read_manifest_raw(execution_id))

    def read_artifact(self, execution_id: str, filename: str) -> str:
        if not _ARTIFACT_NAME_RE.match(filename) or filename == _MANIFEST_FILENAME:
            raise ArtifactRepositoryError(f"invalid artifact filename: {filename!r}")
        path = self._run_dir(execution_id) / filename
        if not path.exists():
            raise ArtifactNotFoundError(f"artifact '{filename}' not found in execution '{execution_id}'")
        return path.read_text(encoding="utf-8")

    def verify(self, execution_id: str) -> None:
        """Recompute every artifact hash and compare with the manifest.

        A manifest that cannot be read or parsed is an integrity failure, not
        a crash: the qualification gate chain must be able to turn it into a
        structured REJECT instead of letting a ``ValueError`` escape.
        """
        try:
            manifest = self.load_manifest(execution_id)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise ArtifactIntegrityError(
                f"manifest of execution '{execution_id}' is unreadable or malformed: {exc}"
            ) from exc
        for name, expected in manifest.artifacts.items():
            try:
                actual = sha256_of(self.read_artifact(execution_id, name))
            except (ArtifactNotFoundError, ArtifactRepositoryError) as exc:
                raise ArtifactIntegrityError(f"artifact '{name}' of execution '{execution_id}' is missing") from exc
            if actual != expected:
                raise ArtifactIntegrityError(
                    f"artifact '{name}' of execution '{execution_id}' was modified: "
                    f"manifest {expected} != actual {actual}"
                )

    # -- History -----------------------------------------------------------

    def list_runs(self) -> list[RunArtifactManifest]:
        """All completed runs, newest first."""
        runs: list[RunArtifactManifest] = []
        if not self._root.exists():
            return runs
        for path in self._root.iterdir():
            if not path.is_dir() or path.name.startswith("."):
                continue
            manifest_path = path / _MANIFEST_FILENAME
            if not manifest_path.exists():
                continue
            runs.append(RunArtifactManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8"))))
        runs.sort(key=lambda m: m.completed_at or m.created_at, reverse=True)
        return runs

    def load_behavior_snapshot(self, execution_id: str) -> BehaviorRunSnapshot | None:
        """Load a run's behavior snapshot with artifact-integrity verification.

        The manifest's recorded hash is the single source of truth for the
        historical artifact's real SHA — it is never recomputed from a
        re-serialized snapshot.  Raises ``ArtifactIntegrityError`` when the
        artifact is missing, hash-mismatched, or unparseable; returns None
        only when the run legitimately has no snapshot.
        """
        manifest = self.load_manifest(execution_id)
        expected_sha = manifest.artifacts.get("behavior_snapshot.json")
        if expected_sha is None:
            return None
        try:
            raw = self.read_artifact(execution_id, "behavior_snapshot.json")
        except ArtifactNotFoundError as exc:
            raise ArtifactIntegrityError(
                f"behavior snapshot of execution '{execution_id}' is listed in the manifest but missing on disk"
            ) from exc
        actual_sha = sha256_of(raw)
        if actual_sha != expected_sha:
            raise ArtifactIntegrityError(
                f"behavior snapshot of execution '{execution_id}' was modified: "
                f"manifest {expected_sha} != actual {actual_sha}"
            )
        try:
            return BehaviorRunSnapshot.model_validate(json.loads(raw))
        except ValueError as exc:
            raise ArtifactIntegrityError(
                f"behavior snapshot of execution '{execution_id}' is corrupted and cannot be parsed: {exc}"
            ) from exc

    def load_capability_profile(self, execution_id: str) -> CapabilityProfile | None:
        """Load a run's capability profile with artifact-integrity verification.

        Returns None when the run has no capability_profile.json artifact.
        """
        manifest = self.load_manifest(execution_id)
        expected_sha = manifest.artifacts.get("capability_profile.json")
        if expected_sha is None:
            return None
        try:
            raw = self.read_artifact(execution_id, "capability_profile.json")
        except ArtifactNotFoundError as exc:
            raise ArtifactIntegrityError(
                f"capability profile of execution '{execution_id}' is listed in the manifest but missing on disk"
            ) from exc
        actual_sha = sha256_of(raw)
        if actual_sha != expected_sha:
            raise ArtifactIntegrityError(
                f"capability profile of execution '{execution_id}' was modified: "
                f"manifest {expected_sha} != actual {actual_sha}"
            )
        try:
            from llmtrace.scoring.models import CapabilityProfile

            return CapabilityProfile.model_validate_json(raw)
        except ValueError as exc:
            raise ArtifactIntegrityError(
                f"capability profile of execution '{execution_id}' is corrupted and cannot be parsed: {exc}"
            ) from exc

    def find_behavior_snapshot_candidates(
        self,
        *,
        target_id: str,
        candidate_model_id: str,
        exclude_execution_id: str,
    ) -> list[RunArtifactManifest]:
        """Historical manifests for the same target/model that declare a snapshot.

        Integrity is NOT judged here — the caller loads each candidate via
        :meth:`load_behavior_snapshot` (which verifies the manifest SHA) so
        it can record integrity warnings and keep searching older candidates.
        """
        candidates: list[RunArtifactManifest] = []
        for manifest in self.list_runs():
            if manifest.execution_id == exclude_execution_id:
                continue
            if manifest.target_id != target_id or manifest.candidate_model_id != candidate_model_id:
                continue
            if "behavior_snapshot.json" in manifest.artifacts:
                candidates.append(manifest)
        return candidates

    def find_behavior_snapshots(
        self,
        *,
        target_id: str,
        candidate_model_id: str,
        exclude_execution_id: str,
    ) -> list[tuple[RunArtifactManifest, BehaviorRunSnapshot]]:
        """Verified historical (manifest, snapshot) pairs, newest first.

        The current execution is always excluded — a run must never be compared
        against itself.  Integrity is verified against the manifest SHA;
        corrupted candidates are skipped.  Compatibility is NOT judged here;
        :class:`BehaviorDriftEngine` stays the sole compatibility authority.
        """
        results: list[tuple[RunArtifactManifest, BehaviorRunSnapshot]] = []
        for manifest in self.find_behavior_snapshot_candidates(
            target_id=target_id,
            candidate_model_id=candidate_model_id,
            exclude_execution_id=exclude_execution_id,
        ):
            try:
                snapshot = self.load_behavior_snapshot(manifest.execution_id)
            except ArtifactIntegrityError:
                continue
            if snapshot is not None:
                results.append((manifest, snapshot))
        return results

    # -- Preflight -----------------------------------------------------------

    def ensure_writable(self) -> None:
        """Probe that the artifact root can be created and written to.

        Creates a temporary probe file under the runs root, flushes it to
        disk, and removes it — no formal run artifact is ever produced.
        """
        self._root.mkdir(parents=True, exist_ok=True)
        probe = self._root / f".write-probe-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        try:
            with open(probe, "w", encoding="utf-8") as handle:
                handle.write("probe")
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            probe.unlink(missing_ok=True)
