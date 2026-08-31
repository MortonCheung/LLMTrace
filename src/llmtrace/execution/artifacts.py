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
import re
import shutil
from pathlib import Path

from llmtrace.analysis.behavior_models import BehaviorRunSnapshot
from llmtrace.execution.models import MANIFEST_VERSION, RunArtifactManifest

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

    def read_artifact(self, execution_id: str, filename: str) -> str:
        if not _ARTIFACT_NAME_RE.match(filename) or filename == _MANIFEST_FILENAME:
            raise ArtifactRepositoryError(f"invalid artifact filename: {filename!r}")
        path = self._run_dir(execution_id) / filename
        if not path.exists():
            raise ArtifactNotFoundError(f"artifact '{filename}' not found in execution '{execution_id}'")
        return path.read_text(encoding="utf-8")

    def verify(self, execution_id: str) -> None:
        """Recompute every artifact hash and compare with the manifest."""
        manifest = self.load_manifest(execution_id)
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
        """Load a run's behavior snapshot, or None when the run has none."""
        try:
            raw = self.read_artifact(execution_id, "behavior_snapshot.json")
        except ArtifactNotFoundError:
            return None
        return BehaviorRunSnapshot.model_validate(json.loads(raw))

    def find_behavior_snapshots(
        self,
        *,
        target_id: str,
        candidate_model_id: str,
        exclude_execution_id: str,
    ) -> list[tuple[RunArtifactManifest, BehaviorRunSnapshot]]:
        """Historical (manifest, snapshot) pairs for the same target/model, newest first.

        The current execution is always excluded — a run must never be compared
        against itself.  Compatibility is NOT judged here; the repository only
        narrows candidates, and :class:`BehaviorDriftEngine` stays the sole
        compatibility authority.
        """
        results: list[tuple[RunArtifactManifest, BehaviorRunSnapshot]] = []
        for manifest in self.list_runs():
            if manifest.execution_id == exclude_execution_id:
                continue
            if manifest.target_id != target_id or manifest.candidate_model_id != candidate_model_id:
                continue
            snapshot = self.load_behavior_snapshot(manifest.execution_id)
            if snapshot is not None:
                results.append((manifest, snapshot))
        return results
