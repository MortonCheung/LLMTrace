"""Shared fixtures for the v0.4-A reference-layer tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmtrace.execution.artifacts import RunArtifactRepository
from llmtrace.execution.models import RunArtifactManifest
from llmtrace.scoring.reference import ReferenceRepository

from .helpers import DEFAULT_EXECUTION_ID, commit_run


@pytest.fixture
def artifact_root(tmp_path: Path) -> Path:
    """Root of the run artifact repository (``runs/`` lives beneath it)."""
    return tmp_path / "reference-runs"


@pytest.fixture
def reference_dir(tmp_path: Path) -> Path:
    """Root of the reference store (``snapshots/`` and ``sets/`` beneath it)."""
    return tmp_path / "references"


@pytest.fixture
def qualified_run(artifact_root: Path) -> tuple[RunArtifactRepository, RunArtifactManifest]:
    """A committed run artifact that passes every qualification gate."""
    return commit_run(artifact_root)


@pytest.fixture
def snapshot_repository(reference_dir: Path) -> ReferenceRepository:
    """Append-only ReferenceSnapshot store under ``references/snapshots``."""
    return ReferenceRepository(directory=reference_dir / "snapshots")


@pytest.fixture
def saved_snapshot(snapshot_repository: ReferenceRepository) -> str:
    """Persist one trusted snapshot; returns its snapshot_id."""
    from .helpers import make_snapshot

    snapshot_id = "openai-gpt-x-quick-v1"
    snapshot_repository.save(make_snapshot(snapshot_id=snapshot_id))
    return snapshot_id


@pytest.fixture
def execution_id() -> str:
    return DEFAULT_EXECUTION_ID
