"""ReferenceSet auto-discovery (CLI Real-Use Sprint §8–§10).

Users who keep trusted calibration sets at the default location
(``~/.llmtrace/references/sets/``) should not have to pass a long
``--reference-set`` path on every run.  This module scans that directory and
keeps only the sets that actually pass :func:`validate_reference_set_for_calibration`.

Priority (§10)::

    explicit --reference-set  >  auto discovery  >  none

Rules (handbook §9 / §11):
  * 0 compatible sets  → calibration unavailable, run continues
  * exactly 1          → auto-used and announced before the run starts
  * >1                 → never pick at random; the CLI fails closed in
                         non-interactive mode and asks in interactive mode
  * fixture / demo / mock sources can never qualify — the shared validator
    enforces the ``operator_verified_api_run`` provenance gate, so "discovery"
    simply never returns them.

Everything here is read-only: no target HTTP, no API key, no provider, no
artifact writes.  A validator-only pass keeps ``--dry-run`` honest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llmtrace.appdir import default_data_root
from llmtrace.execution.artifacts import RunArtifactRepository
from llmtrace.reference.validation import validate_reference_set_for_calibration

_SETS_DIRNAME = "sets"


@dataclass(frozen=True)
class DiscoveryResolution:
    """Result of resolving the calibration set for one run.

    ``resolved_path`` is set exactly when a compatible ReferenceSet was found
    (explicitly or by discovery).  ``candidates`` carries every compatible set
    when more than one exists, so the CLI can fail closed / ask the user.
    """

    resolved_path: Path | None
    candidates: tuple[Path, ...] = ()


def reference_sets_dir(data_root: Path | None = None) -> Path:
    """Default ReferenceSet directory: ``<data_root>/references/sets``.

    Uses the standard app layout (``$LLMTRACE_HOME`` or ``~/.llmtrace``) so
    Web and CLI share one reference universe (§8).
    """
    root = data_root if data_root is not None else default_data_root()
    return root.resolve() / "references" / _SETS_DIRNAME


def discover_compatible_sets(
    artifact_repository: RunArtifactRepository,
    sets_dir: Path | None = None,
) -> list[Path]:
    """Return statements-level compatible ReferenceSet paths under ``sets_dir``.

    Each candidate is fully verified (self-hash + trust chain + compatibility
    gate) via the shared validator; anything that fails is skipped.  The list
    is returned in sorted path order for deterministic behavior.
    """
    directory = sets_dir if sets_dir is not None else reference_sets_dir()
    if not directory.is_dir():
        return []
    candidates: list[Path] = []
    for set_path in sorted(directory.glob("*.json")):
        if not set_path.is_file():
            continue
        try:
            validate_reference_set_for_calibration(
                set_path=set_path,
                artifact_repository=artifact_repository,
            )
        except Exception:
            # Malformed / untrusted / incompatible sets never qualify for
            # trusted calibration; they are ignored, not fatal (§9, §11).
            continue
        candidates.append(set_path)
    return candidates


def resolve_calibration_set(
    *,
    explicit_path: Path | None,
    artifact_repository: RunArtifactRepository,
    sets_dir: Path | None = None,
) -> DiscoveryResolution:
    """Resolve the calibration set for a run: explicit > discovery > none (§10)."""
    if explicit_path is not None:
        # The validator is the single source of truth; an explicit path that
        # cannot support formal calibration is a user-facing error, raised now
        # (identical to the runner's preflight).
        validate_reference_set_for_calibration(
            set_path=explicit_path,
            artifact_repository=artifact_repository,
        )
        return DiscoveryResolution(resolved_path=explicit_path)

    candidates = discover_compatible_sets(artifact_repository, sets_dir=sets_dir)
    if len(candidates) == 1:
        return DiscoveryResolution(resolved_path=candidates[0], candidates=(candidates[0],))
    return DiscoveryResolution(resolved_path=None, candidates=tuple(candidates))


__all__: list[str] = [
    "DiscoveryResolution",
    "discover_compatible_sets",
    "reference_sets_dir",
    "resolve_calibration_set",
]
