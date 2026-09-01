"""CLI ``reference`` 子命令测试（§42）.

覆盖：``reference capture --dry-run``（0 HTTP / 0 副作用）、mock provider
happy path（CAPTURED / QUALIFICATION_REJECTED / 缺 API key）、``reference
set-create``（happy path / missing snapshot / incompatible member）。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmtrace.cli import app
from llmtrace.reference import ReferenceCaptureService
from llmtrace.reference.capture import ReferenceCaptureResult, ReferenceCaptureStatus
from llmtrace.scoring.reference import ReferenceRepository
from tests.reference.helpers import DEFAULT_EXECUTION_ID, make_snapshot

runner = CliRunner()

# 新版 Typer/Rich 的帮助输出可能包含 ANSI 控制符（尤其在 CI 环境），
# 断言前需剥离（等效 click.utils.strip_ansi；click 已不再是 typer 的依赖）。
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    """剥离 ANSI 控制序列，返回纯文本."""
    return _ANSI_ESCAPE.sub("", text)


def _capture_args(
    *,
    reference_dir: Path,
    output_dir: Path,
    snapshot_id: str = "operator-test-snapshot",
    api_key_env: str = "LLMTRACE_TEST_KEY",
) -> list[str]:
    """公共 reference capture 参数（dry-run 或 live 共用）."""
    return [
        "reference",
        "capture",
        "--protocol",
        "openai",
        "--base-url",
        "http://localhost:9999/v1",
        "--model",
        "my-model",
        "--api-key-env",
        api_key_env,
        "--provider-id",
        "operator-test",
        "--snapshot-id",
        snapshot_id,
        "--created-by",
        "operator",
        "--reference-dir",
        str(reference_dir),
        "--output-dir",
        str(output_dir),
    ]


def _save_snapshots(reference_dir: Path, *snapshots: object) -> ReferenceRepository:
    """Persist snapshots under ``<reference_dir>/snapshots`` and return the repo.

    Uses save_trusted() to write both snapshot.json and sidecar, making them
    eligible for trusted ReferenceSet creation.
    """
    repo = ReferenceRepository(directory=reference_dir / "snapshots")
    for snapshot in snapshots:
        repo.save_trusted(snapshot)  # type: ignore[arg-type]
    return repo


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------


class TestReferenceHelp:
    def test_capture_help(self) -> None:
        result = runner.invoke(app, ["reference", "capture", "--help"])
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 0
        assert "--protocol" in stdout
        assert "--base-url" in stdout
        assert "--api-key-env" in stdout
        assert "--snapshot-id" in stdout
        assert "--dry-run" in stdout

    def test_set_create_help(self) -> None:
        result = runner.invoke(app, ["reference", "set-create", "--help"])
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 0
        assert "--set-id" in stdout
        assert "--set-version" in stdout
        assert "--snapshot" in stdout


# ---------------------------------------------------------------------------
# reference capture --dry-run
# ---------------------------------------------------------------------------


class TestCaptureDryRun:
    def test_dry_run_prints_plan_with_suite_content_sha(self, tmp_path: Path) -> None:
        """Dry run: prints the plan (incl. Suite Content SHA), exits 0, no side effects."""
        reference_dir = tmp_path / "references"
        output_dir = tmp_path / "reference-runs"
        result = runner.invoke(
            app,
            _capture_args(reference_dir=reference_dir, output_dir=output_dir) + ["--dry-run"],
        )
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 0, stdout
        assert "Suite Content SHA" in stdout
        # Rich 表格会省略长 hex，只要求至少 32 位可见前缀。
        sha = re.search(r"[0-9a-f]{32}", stdout)
        assert sha is not None, f"expected a 64-hex suite content SHA in output: {stdout!r}"
        # 0 HTTP / 0 artifact / 0 snapshot — nothing may be written.
        assert not (output_dir / "runs").exists()
        assert not (reference_dir / "snapshots").exists()


# ---------------------------------------------------------------------------
# reference capture (live, mock provider)
# ---------------------------------------------------------------------------


class TestCaptureLive:
    def test_captured_saves_snapshot_message(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        async def fake_capture(self: object, **kwargs: object) -> ReferenceCaptureResult:
            return ReferenceCaptureResult(
                execution_id=DEFAULT_EXECUTION_ID,
                status=ReferenceCaptureStatus.CAPTURED,
                snapshot_id="operator-test-snapshot",
            )

        monkeypatch.setenv("LLMTRACE_TEST_KEY", "sk-test-secret")
        monkeypatch.setattr(ReferenceCaptureService, "capture", fake_capture)
        reference_dir = tmp_path / "references"
        output_dir = tmp_path / "reference-runs"
        result = runner.invoke(
            app,
            _capture_args(reference_dir=reference_dir, output_dir=output_dir) + ["--yes"],
        )
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 0, stdout
        assert "[OK] ReferenceSnapshot 已保存" in stdout
        assert DEFAULT_EXECUTION_ID in stdout

    def test_qualification_rejected_exits_1(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        async def fake_capture(self: object, **kwargs: object) -> ReferenceCaptureResult:
            return ReferenceCaptureResult(
                execution_id=DEFAULT_EXECUTION_ID,
                status=ReferenceCaptureStatus.QUALIFICATION_REJECTED,
                reason_codes=("INCOMPLETE_MEASUREMENT",),
                warnings=("4 failures",),
            )

        monkeypatch.setenv("LLMTRACE_TEST_KEY", "sk-test-secret")
        monkeypatch.setattr(ReferenceCaptureService, "capture", fake_capture)
        result = runner.invoke(
            app,
            _capture_args(
                reference_dir=tmp_path / "references",
                output_dir=tmp_path / "reference-runs",
            )
            + ["--yes"],
        )
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 1
        assert "未通过资格门禁" in stdout
        assert "INCOMPLETE_MEASUREMENT" in stdout

    def test_missing_api_key_exits_1(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("LLMTRACE_TEST_KEY", raising=False)
        result = runner.invoke(
            app,
            _capture_args(
                reference_dir=tmp_path / "references",
                output_dir=tmp_path / "reference-runs",
            )
            + ["--yes"],
        )
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 1
        assert "不存在或为空" in stdout


# ---------------------------------------------------------------------------
# CLI-level URL scrub regression
# ---------------------------------------------------------------------------


class TestCLICaptureURLScrubRegression:
    """End-to-end: secrets must never appear in confirm prompt or error output."""

    SECRET_URL = "https://myuser:mypassword@example.com/v1?api_key=secret123&region=us"

    def test_confirm_prompt_scrubs_credentials(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        captured_prompts: list[str] = []

        def capturing_confirm(message: str, *args: object, **kwargs: object) -> bool:
            captured_prompts.append(message)
            return False

        monkeypatch.setenv("LLMTRACE_TEST_KEY", "sk-test-secret")
        monkeypatch.setattr("typer.confirm", capturing_confirm)

        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

        reference_dir = tmp_path / "references"
        output_dir = tmp_path / "reference-runs"
        result = runner.invoke(
            app,
            _capture_args(reference_dir=reference_dir, output_dir=output_dir) + ["--base-url", self.SECRET_URL],
        )
        stdout = _strip_ansi(result.stdout)
        all_text = stdout + " ".join(captured_prompts)

        assert "myuser" not in all_text
        assert "mypassword" not in all_text
        assert "secret123" not in all_text

    def test_exception_output_scrubs_secrets(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        async def fake_capture(self: object, **kwargs: object) -> ReferenceCaptureResult:
            raise RuntimeError("API request failed: myuser:mypassword@evil.com api_key=secret123 query failed")

        monkeypatch.setenv("LLMTRACE_TEST_KEY", "sk-test-secret")
        monkeypatch.setattr(ReferenceCaptureService, "capture", fake_capture)

        reference_dir = tmp_path / "references"
        output_dir = tmp_path / "reference-runs"
        result = runner.invoke(
            app,
            _capture_args(reference_dir=reference_dir, output_dir=output_dir)
            + ["--base-url", self.SECRET_URL]
            + ["--yes"],
        )
        stdout = _strip_ansi(result.stdout)

        assert "myuser" not in stdout
        assert "mypassword" not in stdout
        assert "secret123" not in stdout


# ---------------------------------------------------------------------------
# reference set-create
# ---------------------------------------------------------------------------


class TestSetCreate:
    def test_happy_path_writes_set_file(self, tmp_path: Path) -> None:
        reference_dir = tmp_path / "references"
        _save_snapshots(
            reference_dir,
            make_snapshot(snapshot_id="ref-001"),
            make_snapshot(snapshot_id="ref-002"),
        )
        result = runner.invoke(
            app,
            [
                "reference",
                "set-create",
                "--reference-dir",
                str(reference_dir),
                "--set-id",
                "refset-v1",
                "--set-version",
                "1.0.0",
                "--snapshot",
                "ref-001",
                "--snapshot",
                "ref-002",
                "--description",
                "trusted reference set",
            ],
        )
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 0, stdout
        assert "[OK] ReferenceSet 已保存" in stdout
        assert (reference_dir / "sets" / "refset-v1_1.0.0.json").is_file()

    def test_missing_snapshot_exits_1(self, tmp_path: Path) -> None:
        reference_dir = tmp_path / "references"
        result = runner.invoke(
            app,
            [
                "reference",
                "set-create",
                "--reference-dir",
                str(reference_dir),
                "--set-id",
                "refset-v1",
                "--set-version",
                "1.0.0",
                "--snapshot",
                "does-not-exist",
            ],
        )
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 1
        assert "not found" in stdout.lower() or "不存在" in stdout

    def test_incompatible_member_exits_1(self, tmp_path: Path) -> None:
        reference_dir = tmp_path / "references"
        _save_snapshots(
            reference_dir,
            make_snapshot(snapshot_id="ref-001"),
            make_snapshot(snapshot_id="ref-002", adapter_id="other-adapter-v2"),
        )
        result = runner.invoke(
            app,
            [
                "reference",
                "set-create",
                "--reference-dir",
                str(reference_dir),
                "--set-id",
                "refset-v1",
                "--set-version",
                "1.0.0",
                "--snapshot",
                "ref-001",
                "--snapshot",
                "ref-002",
            ],
        )
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 1
        assert "incompatible" in stdout.lower() or "不兼容" in stdout
