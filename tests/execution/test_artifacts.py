"""Tests for the append-only RunArtifactRepository (execution/artifacts.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmtrace.execution.artifacts import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ArtifactRepositoryError,
    DuplicateExecutionError,
    RunArtifactRepository,
)

from .conftest import make_manifest

EXEC_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
ARTIFACTS = {"report.json": '{"ok": true}', "report.html": "<html></html>"}


class TestRunArtifactRepository:
    def test_commit_writes_manifest_last_and_hashes(self, tmp_path: Path) -> None:
        repo = RunArtifactRepository(tmp_path)
        manifest = make_manifest(execution_id=EXEC_ID)
        final = repo.commit(manifest, dict(ARTIFACTS))

        run_dir = tmp_path / "runs" / EXEC_ID
        assert run_dir.is_dir()
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "report.json").exists()

        # Manifest contains artifact hashes and was written with them recorded.
        assert set(final.artifacts) == {"report.json", "report.html"}
        assert final.artifacts["report.json"] is not None

    def test_duplicate_execution_rejected(self, tmp_path: Path) -> None:
        repo = RunArtifactRepository(tmp_path)
        repo.commit(make_manifest(execution_id=EXEC_ID), dict(ARTIFACTS))
        with pytest.raises(DuplicateExecutionError):
            repo.commit(make_manifest(execution_id=EXEC_ID), dict(ARTIFACTS))

    def test_verify_detects_corruption(self, tmp_path: Path) -> None:
        repo = RunArtifactRepository(tmp_path)
        repo.commit(make_manifest(execution_id=EXEC_ID), dict(ARTIFACTS))

        # Tamper with an artifact after commit.
        (tmp_path / "runs" / EXEC_ID / "report.json").write_text('{"ok": false}')
        with pytest.raises(ArtifactIntegrityError):
            repo.verify(EXEC_ID)

    def test_verify_passes_when_unchanged(self, tmp_path: Path) -> None:
        repo = RunArtifactRepository(tmp_path)
        repo.commit(make_manifest(execution_id=EXEC_ID), dict(ARTIFACTS))
        repo.verify(EXEC_ID)

    def test_path_traversal_rejected(self, tmp_path: Path) -> None:
        repo = RunArtifactRepository(tmp_path)
        for bad in ("../evil", "a/b", "..", "."):
            with pytest.raises(ArtifactRepositoryError):
                repo._run_dir(bad)

    def test_invalid_artifact_filename_rejected(self, tmp_path: Path) -> None:
        repo = RunArtifactRepository(tmp_path)
        with pytest.raises(ArtifactRepositoryError):
            repo.commit(make_manifest(execution_id=EXEC_ID), {"../evil.json": "x"})
        with pytest.raises(ArtifactRepositoryError):
            repo.commit(make_manifest(execution_id=EXEC_ID), {"manifest.json": "x"})

    def test_empty_artifacts_rejected(self, tmp_path: Path) -> None:
        repo = RunArtifactRepository(tmp_path)
        with pytest.raises(ArtifactRepositoryError):
            repo.commit(make_manifest(execution_id=EXEC_ID), {})

    def test_load_manifest_and_read_artifact(self, tmp_path: Path) -> None:
        repo = RunArtifactRepository(tmp_path)
        repo.commit(make_manifest(execution_id=EXEC_ID), dict(ARTIFACTS))
        loaded = repo.load_manifest(EXEC_ID)
        assert loaded.execution_id == EXEC_ID
        assert repo.read_artifact(EXEC_ID, "report.json") == ARTIFACTS["report.json"]

    def test_load_missing_manifest_raises(self, tmp_path: Path) -> None:
        repo = RunArtifactRepository(tmp_path)
        with pytest.raises(ArtifactNotFoundError):
            repo.load_manifest(EXEC_ID)

    def test_list_runs_newest_first(self, tmp_path: Path) -> None:
        repo = RunArtifactRepository(tmp_path)
        ids = [
            "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeee1",
            "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeee2",
        ]
        repo.commit(make_manifest(execution_id=ids[0]), dict(ARTIFACTS))
        repo.commit(make_manifest(execution_id=ids[1]), dict(ARTIFACTS))
        runs = repo.list_runs()
        assert [r.execution_id for r in runs] == [ids[1], ids[0]]

    def test_write_failure_rolls_back(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        repo = RunArtifactRepository(tmp_path)
        # Force the atomic rename to fail; the staging dir must be cleaned up
        # and no final run dir may appear.
        import pathlib

        original = pathlib.Path.rename

        def _boom(self: Path, target: object) -> None:  # noqa: ANN001
            raise OSError("simulated rename failure")

        monkeypatch.setattr(pathlib.Path, "rename", _boom)
        with pytest.raises(OSError):
            repo.commit(make_manifest(execution_id=EXEC_ID), dict(ARTIFACTS))

        monkeypatch.setattr(pathlib.Path, "rename", original)
        # No half-written final run directory.
        assert not (tmp_path / "runs" / EXEC_ID).exists()
        # No leftover staging directory.
        staging = list((tmp_path / "runs").glob(".staging-*"))
        assert staging == []
