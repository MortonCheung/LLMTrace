"""Fingerprint reference set auto-discovery (Task 36).

Priority is pinned in both directions: an explicit ``--fingerprint-set`` always
wins, a single compatible set is adopted, and two compatible sets are never
chosen between — the resolver returns no path and hands every candidate back so
the caller can fail closed.  The compatibility gate is the same offline
``resolve_fingerprint_context`` the runner's preflight uses, so a set captured
with different repetitions (or with a broken self-hash) simply never becomes a
candidate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llmtrace.fingerprint.discovery import (
    discover_compatible_sets,
    resolve_fingerprint_set,
)
from llmtrace.fingerprint.models import FingerprintProfile, FingerprintSuite, repetitions_for_profile
from llmtrace.fingerprint.repository import FingerprintRepository
from llmtrace.fingerprint.runtime import FingerprintRuntimeError

from .conftest import MODEL_ID, make_reference_snapshot, publish_reference_set

_STANDARD = repetitions_for_profile(FingerprintProfile.STANDARD)
_QUICK = repetitions_for_profile(FingerprintProfile.QUICK)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _data_root(tmp_path: Path) -> Path:
    return tmp_path / "appdata"


def _repository(tmp_path: Path) -> FingerprintRepository:
    return FingerprintRepository.load(data_root=_data_root(tmp_path))


def _publish(
    tmp_path: Path,
    suite: FingerprintSuite,
    repetitions: int,
    *,
    set_id: str,
    snapshot_id: str,
    choice_index: int = 0,
) -> Path:
    """Publish one single-capture reference set and return its on-disk path."""
    snapshot = make_reference_snapshot(
        suite=suite,
        snapshot_id=snapshot_id,
        model_id=MODEL_ID,
        repetitions=repetitions,
        choice_index=choice_index,
    )
    fixture = publish_reference_set(
        data_root=_data_root(tmp_path),
        suite=suite,
        snapshots=(snapshot,),
        set_id=set_id,
    )
    return fixture.set_path


def _sets_dir(tmp_path: Path) -> Path:
    return _data_root(tmp_path) / "fingerprints" / "sets"


# ---------------------------------------------------------------------------
# Resolution priority
# ---------------------------------------------------------------------------


class TestResolutionPriority:
    def test_explicit_path_wins_without_scanning(self, tmp_path: Path, suite: FingerprintSuite) -> None:
        set_path = _publish(tmp_path, suite, _STANDARD, set_id="set-a", snapshot_id="snap-a")

        resolution = resolve_fingerprint_set(
            explicit_path=set_path,
            repository=_repository(tmp_path),
            repetitions=_STANDARD,
        )

        assert resolution.resolved_path == set_path
        assert resolution.candidates == ()

    def test_explicit_unreadable_path_is_fatal(self, tmp_path: Path, suite: FingerprintSuite) -> None:
        # An explicit path is a user assertion, so an unusable set is an error
        # rather than a silent fallback to discovery.
        with pytest.raises(FingerprintRuntimeError, match="unreadable"):
            resolve_fingerprint_set(
                explicit_path=tmp_path / "missing.json",
                repository=_repository(tmp_path),
                repetitions=_STANDARD,
            )

    def test_absent_sets_directory_resolves_to_none(self, tmp_path: Path) -> None:
        resolution = resolve_fingerprint_set(
            explicit_path=None,
            repository=_repository(tmp_path),
            repetitions=_STANDARD,
        )

        assert resolution.resolved_path is None
        assert resolution.candidates == ()

    def test_single_compatible_set_is_discovered(self, tmp_path: Path, suite: FingerprintSuite) -> None:
        _publish(tmp_path, suite, _STANDARD, set_id="set-a", snapshot_id="snap-a")

        resolution = resolve_fingerprint_set(
            explicit_path=None,
            repository=_repository(tmp_path),
            repetitions=_STANDARD,
        )

        assert resolution.resolved_path is not None
        assert resolution.resolved_path.parent == _sets_dir(tmp_path)
        assert resolution.candidates == (resolution.resolved_path,)

    def test_multiple_compatible_sets_fail_closed(self, tmp_path: Path, suite: FingerprintSuite) -> None:
        _publish(tmp_path, suite, _STANDARD, set_id="set-a", snapshot_id="snap-a")
        _publish(tmp_path, suite, _STANDARD, set_id="set-b", snapshot_id="snap-b")

        resolution = resolve_fingerprint_set(
            explicit_path=None,
            repository=_repository(tmp_path),
            repetitions=_STANDARD,
        )

        # Never pick at random: no path is resolved and every candidate is
        # reported so the CLI can demand --fingerprint-set.
        assert resolution.resolved_path is None
        assert len(resolution.candidates) == 2
        assert resolution.candidates == tuple(sorted(resolution.candidates))


# ---------------------------------------------------------------------------
# Compatibility gate
# ---------------------------------------------------------------------------


class TestCompatibilityGate:
    def test_set_captured_with_other_repetitions_is_skipped(self, tmp_path: Path, suite: FingerprintSuite) -> None:
        _publish(tmp_path, suite, _QUICK, set_id="set-quick", snapshot_id="snap-quick")

        resolution = resolve_fingerprint_set(
            explicit_path=None,
            repository=_repository(tmp_path),
            repetitions=_STANDARD,
        )

        assert resolution.resolved_path is None
        assert resolution.candidates == ()

    def test_tampered_self_hash_is_never_a_candidate(self, tmp_path: Path, suite: FingerprintSuite) -> None:
        set_path = _publish(tmp_path, suite, _STANDARD, set_id="set-a", snapshot_id="snap-a")
        tampered = set_path.read_text(encoding="utf-8").replace('"set-a"', '"set-a-tampered"')
        (_sets_dir(tmp_path) / "set-a-tampered_0.1.0.json").write_text(tampered, encoding="utf-8")

        resolution = resolve_fingerprint_set(
            explicit_path=None,
            repository=_repository(tmp_path),
            repetitions=_STANDARD,
        )

        # The copied set no longer matches its declared content hash, so the
        # remaining honest set is the only candidate.
        assert resolution.resolved_path is not None
        assert resolution.resolved_path.name == "set-a_0.1.0.json"
        assert len(resolution.candidates) == 1

    def test_foreign_files_are_ignored(self, tmp_path: Path, suite: FingerprintSuite) -> None:
        _publish(tmp_path, suite, _STANDARD, set_id="set-a", snapshot_id="snap-a")
        (_sets_dir(tmp_path) / "notes.txt").write_text("ignored", encoding="utf-8")

        resolution = resolve_fingerprint_set(
            explicit_path=None,
            repository=_repository(tmp_path),
            repetitions=_STANDARD,
        )

        assert resolution.resolved_path is not None
        assert resolution.resolved_path.name == "set-a_0.1.0.json"

    def test_set_with_a_missing_member_snapshot_is_skipped(self, tmp_path: Path, suite: FingerprintSuite) -> None:
        _publish(tmp_path, suite, _STANDARD, set_id="set-a", snapshot_id="snap-a")
        # The set still parses and self-verifies, but one of its members is no
        # longer in the snapshot repository — that set cannot support matching.
        (_data_root(tmp_path) / "fingerprints" / "snapshots" / "snap-a.json").unlink()

        resolution = resolve_fingerprint_set(
            explicit_path=None,
            repository=_repository(tmp_path),
            repetitions=_STANDARD,
        )

        assert resolution.resolved_path is None
        assert resolution.candidates == ()

    def test_discover_honours_sets_dir_override(self, tmp_path: Path, suite: FingerprintSuite) -> None:
        _publish(tmp_path, suite, _STANDARD, set_id="set-a", snapshot_id="snap-a")
        empty_dir = tmp_path / "elsewhere"
        empty_dir.mkdir()

        repository = _repository(tmp_path)

        assert discover_compatible_sets(repository, repetitions=_STANDARD, sets_dir=empty_dir) == []
        assert len(discover_compatible_sets(repository, repetitions=_STANDARD)) == 1

    def test_repetitions_below_one_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="repetitions must be >= 1"):
            resolve_fingerprint_set(
                explicit_path=None,
                repository=_repository(tmp_path),
                repetitions=0,
            )
