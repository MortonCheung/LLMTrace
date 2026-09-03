"""CLI ``run --reference-set`` tests — shared validator + zero side effects.

§12: the CLI dry-run and the runner preflight must use the *same*
ReferenceSet validator, so an untrusted or incompatible set is refused by
both, before any request.

§13: a dry-run may read the ReferenceSet, snapshot sidecars, source run
manifests and capability profiles — but must leave the artifact store
untouched (0 target HTTP, 0 provider, 0 sandbox, 0 committed run).
"""

from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from llmtrace.cli import app
from tests.reference.helpers import build_trusted_reference_root

runner = CliRunner()

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


def _run_args(output_dir: Path, set_path: Path) -> list[str]:
    return [
        "run",
        "--protocol",
        "openai",
        "--base-url",
        "http://test.example.com/v1",
        "--model",
        "demo-model",
        "--api-key-env",
        "TEST_API_KEY",
        "-o",
        str(output_dir),
        "--reference-set",
        str(set_path),
        "--dry-run",
        "--yes",
    ]


def _committed_runs(output_dir: Path) -> set[str]:
    runs_dir = output_dir / "runs"
    if not runs_dir.exists():
        return set()
    return {p.name for p in runs_dir.iterdir() if p.is_dir() and not p.name.startswith(".")}


class TestRunDryRunReferenceSet:
    def test_dry_run_prints_verified_calibration_context(self, tmp_path: Path) -> None:
        """A real trust chain validates, and dry-run reports the context."""
        output_dir = tmp_path / "ref"
        _ref_root, set_path = build_trusted_reference_root(output_dir)
        before = _committed_runs(output_dir)

        result = runner.invoke(app, _run_args(output_dir, set_path))
        stdout = _strip_ansi(result.stdout)

        assert result.exit_code == 0, stdout
        assert "Reference Calibration" in stdout
        assert "calib-set" in stdout
        assert "llmtrace-reference-calibration-v1" in stdout

        # §13: reading is allowed, committing is not.
        assert _committed_runs(output_dir) == before

    def test_dry_run_rejects_forged_set_without_side_effects(self, tmp_path: Path) -> None:
        """A forged set (valid self-hash, wrong member SHA) fails the shared
        validator at dry-run time — before any request — and writes nothing."""
        output_dir = tmp_path / "ref"
        _ref_root, set_path = build_trusted_reference_root(output_dir)

        from llmtrace.reference.reference_set import ReferenceSet

        data = ReferenceSet.model_validate_json(set_path.read_text(encoding="utf-8")).model_dump()
        data["members"][0]["snapshot_sha256"] = "9" * 64
        forged = ReferenceSet.model_validate(data).model_copy(
            update={"content_sha256": ReferenceSet.model_validate(data).compute_content_sha256()}
        )
        forged_path = tmp_path / "ref" / "references" / "sets" / "forged.json"
        forged_path.write_text(forged.model_dump_json())

        before = _committed_runs(output_dir)
        result = runner.invoke(app, _run_args(output_dir, forged_path))
        stdout = _strip_ansi(result.stdout)

        assert result.exit_code == 1
        assert "reference set" in stdout.lower()
        # Refused before the request would ever be sent, and nothing written.
        assert _committed_runs(output_dir) == before
