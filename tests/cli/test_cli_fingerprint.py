"""CLI ``fingerprint`` 子命令测试（Task 37 / 38 / 39）.

覆盖：``fingerprint --help`` / ``capture --help``（含"没有裸 ``--api-key``
flag"这条 Task 38 纪律）、respx mock 下的真实采集路径（Rule 4：请求经由
Provider → RequestBudget → EvidenceRecorder）、Task 39 完成输出、缺 key /
非交互未确认 / 非法 role / 非法 profile / 非法 snapshot id / 重复 snapshot
的 fail-closed 行为，以及错误输出与 confirm 提示的秘密清洗。
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from llmtrace.cli import app
from llmtrace.fingerprint.executor import FingerprintExecutor
from llmtrace.fingerprint.models import (
    FingerprintProfile,
    FingerprintSourceRole,
    repetitions_for_profile,
)
from llmtrace.fingerprint.repository import FingerprintRepository
from llmtrace.fingerprint.suite import load_fingerprint_suite
from tests.fingerprint.conftest import BASE_URL, MODEL_ID, mock_openai

runner = CliRunner()


class _TtyBytesIO(io.BytesIO):
    """供 ``CliRunner(input=...)`` 使用的伪 TTY 输入流.

    ``CliRunner`` 会把传入的 ``input`` 包进 ``_NamedTextIOWrapper``，其
    ``isatty()`` 委托给底层 buffer；默认 ``BytesIO`` 返回 False，于是
    ``fingerprint capture`` 必然走"非交互式需要 --yes"分支。覆写 ``isatty``
    即可让测试真正走到交互式确认分支。
    """

    def isatty(self) -> bool:
        return True


def _tty_bytes_io(_input: object, _charset: str) -> io.BytesIO:
    """``typer.testing.make_input_stream`` 的替身：返回 isatty() 为 True 的流."""
    return _TtyBytesIO(b"")


_SECRET = "sk-fingerprint-secret"
_SECRET_URL = "https://myuser:mypassword@test.example.com/v1?api_key=secret123&region=us"

# 新版 Typer/Rich 的帮助输出可能包含 ANSI 控制符（尤其在 CI 环境），
# 断言前需剥离（等效 click.utils.strip_ansi）。
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    """剥离 ANSI 控制序列，返回纯文本."""
    return _ANSI_ESCAPE.sub("", text)


def _compact(text: str) -> str:
    """去掉所有空白，便于跨 Rich 折行做子串断言."""
    return re.sub(r"\s+", "", text)


def _capture_args(
    *,
    data_dir: Path,
    snapshot_id: str = "fixture-official-baseline",
    base_url: str = BASE_URL,
    api_key_env: str = "LLMTRACE_TEST_KEY",
    extra: list[str] | None = None,
) -> list[str]:
    """公共 ``fingerprint capture`` 参数（--yes 已在其中，测试无需交互）."""
    args = [
        "fingerprint",
        "capture",
        "--base-url",
        base_url,
        "--model",
        MODEL_ID,
        "--role",
        "official-baseline",
        "--profile",
        "quick",
        "--api-key-env",
        api_key_env,
        "--snapshot-id",
        snapshot_id,
        "--data-dir",
        str(data_dir),
        "--yes",
    ]
    if extra is not None:
        args.extend(extra)
    return args


def _drop_option(args: list[str], *flags: str) -> list[str]:
    """移除 ``--flag value`` 形式的选项，让默认值/自动推导分支生效."""
    kept: list[str] = []
    index = 0
    while index < len(args):
        if args[index] in flags:
            index += 2
            continue
        kept.append(args[index])
        index += 1
    return kept


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------


class TestFingerprintHelp:
    def test_group_lists_capture(self) -> None:
        result = runner.invoke(app, ["fingerprint", "--help"])
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 0, stdout
        assert "capture" in stdout

    def test_capture_help_exposes_env_key_not_raw_key(self) -> None:
        result = runner.invoke(app, ["fingerprint", "capture", "--help"])
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 0, stdout
        for flag in ("--base-url", "--model", "--role", "--profile", "--api-key-env", "--snapshot-id"):
            assert flag in stdout
        # Task 38: 明文 key 绝不作为命令行参数出现（只有 *-env 形式）。
        assert re.search(r"--api-key(?![-\w])", stdout) is None


# ---------------------------------------------------------------------------
# capture (live path under respx)
# ---------------------------------------------------------------------------


class TestCaptureLive:
    def test_capture_saves_snapshot_and_reports_task39_shape(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        suite = load_fingerprint_suite()
        data_dir = tmp_path / "appdata"
        monkeypatch.setenv("LLMTRACE_TEST_KEY", _SECRET)

        with respx.mock(assert_all_called=False) as mock:
            mock_openai(mock, suite=suite, choice_index=0)
            result = runner.invoke(app, _capture_args(data_dir=data_dir))

        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 0, stdout

        samples = len(suite.probes) * repetitions_for_profile(FingerprintProfile.QUICK)
        assert "Fingerprint capture complete" in stdout
        assert MODEL_ID in stdout
        assert "operator-asserted official baseline" in stdout
        assert f"{suite.suite_id} {suite.suite_version}" in stdout
        assert f"{samples} / {samples}" in stdout

        snapshot_path = data_dir / "fingerprints" / "snapshots" / "fixture-official-baseline.json"
        assert snapshot_path.is_file()
        assert _compact(str(snapshot_path)) in _compact(stdout)
        assert "fingeset-create" not in _compact(stdout)
        assert "fingerprintset-create" in _compact(stdout)

        repository = FingerprintRepository.load(data_root=data_dir)
        snapshot = repository.snapshots.get("fixture-official-baseline")
        repository.snapshots.verify("fixture-official-baseline")
        assert snapshot.model_id == MODEL_ID
        assert snapshot.provider_id == "openai"
        assert snapshot.source_role is FingerprintSourceRole.OFFICIAL_BASELINE
        assert snapshot.repetitions == repetitions_for_profile(FingerprintProfile.QUICK)
        assert len(snapshot.observations) == samples
        assert snapshot.distributions

        # Task 46 preview: the API key never reaches the persisted snapshot.
        assert _SECRET not in snapshot_path.read_text(encoding="utf-8")

    def test_snapshot_id_is_derived_when_absent(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """省略 ``--snapshot-id`` 时自动推导出 filename-safe 标识，不覆盖已有 snapshot."""
        suite = load_fingerprint_suite()
        data_dir = tmp_path / "appdata"
        monkeypatch.setenv("LLMTRACE_TEST_KEY", _SECRET)

        args = _drop_option(_capture_args(data_dir=data_dir), "--snapshot-id")
        with respx.mock(assert_all_called=False) as mock:
            mock_openai(mock, suite=suite, choice_index=0)
            result = runner.invoke(app, args)

        assert result.exit_code == 0, _strip_ansi(result.stdout)
        snapshots_dir = data_dir / "fingerprints" / "snapshots"
        written = sorted(snapshots_dir.glob("*.json"))
        assert len(written) == 1
        assert re.fullmatch(r"my-real-model-official-baseline-quick-\d{8}T\d{6}Z\.json", written[0].name)

    def test_duplicate_snapshot_id_is_rejected(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        suite = load_fingerprint_suite()
        data_dir = tmp_path / "appdata"
        monkeypatch.setenv("LLMTRACE_TEST_KEY", _SECRET)

        with respx.mock(assert_all_called=False) as mock:
            mock_openai(mock, suite=suite, choice_index=0)
            first = runner.invoke(app, _capture_args(data_dir=data_dir))
            second = runner.invoke(app, _capture_args(data_dir=data_dir))

        assert first.exit_code == 0, _strip_ansi(first.stdout)
        stdout = _strip_ansi(second.stdout)
        assert second.exit_code == 1
        assert "already exists" in stdout

    def test_unsafe_snapshot_id_is_rejected(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        suite = load_fingerprint_suite()
        monkeypatch.setenv("LLMTRACE_TEST_KEY", _SECRET)

        with respx.mock(assert_all_called=False) as mock:
            mock_openai(mock, suite=suite, choice_index=0)
            result = runner.invoke(
                app,
                _capture_args(data_dir=tmp_path / "appdata", snapshot_id="../../escape"),
            )

        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 1
        assert "fingerprint capture" in stdout
        assert not (tmp_path / "escape.json").exists()

    def test_keyboard_interrupt_exits_130(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        async def interrupted_run(self: object, **kwargs: object) -> tuple[object, ...]:
            raise KeyboardInterrupt

        monkeypatch.setenv("LLMTRACE_TEST_KEY", _SECRET)
        monkeypatch.setattr(FingerprintExecutor, "run", interrupted_run)

        with respx.mock(assert_all_called=False) as mock:
            mock_openai(mock, suite=load_fingerprint_suite())
            result = runner.invoke(app, _capture_args(data_dir=tmp_path / "appdata"))

        assert result.exit_code == 130
        assert not (tmp_path / "appdata" / "fingerprints" / "snapshots").exists()

    def test_debug_flag_prints_traceback(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        async def failing_run(self: object, **kwargs: object) -> tuple[object, ...]:
            raise RuntimeError("boom")

        monkeypatch.setenv("LLMTRACE_TEST_KEY", _SECRET)
        monkeypatch.setattr(FingerprintExecutor, "run", failing_run)

        with respx.mock(assert_all_called=False) as mock:
            mock_openai(mock, suite=load_fingerprint_suite())
            result = runner.invoke(
                app,
                _capture_args(data_dir=tmp_path / "appdata", extra=["--debug"]),
            )

        output = _strip_ansi(result.stdout) + _strip_ansi(result.stderr)
        assert result.exit_code == 1
        assert "Traceback (most recent call last)" in output
        assert "RuntimeError: boom" in output


# ---------------------------------------------------------------------------
# fail-closed gates that run before any request
# ---------------------------------------------------------------------------


class TestCaptureGates:
    def test_missing_api_key_exits_1(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("LLMTRACE_TEST_KEY", raising=False)
        result = runner.invoke(app, _capture_args(data_dir=tmp_path / "appdata"))
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 1
        assert "不存在或为空" in stdout

    def test_non_interactive_without_yes_exits_1(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("LLMTRACE_TEST_KEY", _SECRET)
        # CliRunner 的 stdin 默认非 TTY（isatty() -> False），无需额外打桩。
        args = [arg for arg in _capture_args(data_dir=tmp_path / "appdata") if arg != "--yes"]
        result = runner.invoke(app, args)
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 1
        assert "非交互式执行需要 --yes" in stdout
        assert not (tmp_path / "appdata" / "fingerprints").exists()

    def test_unknown_role_is_a_usage_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("LLMTRACE_TEST_KEY", _SECRET)
        result = runner.invoke(
            app,
            _capture_args(data_dir=tmp_path / "appdata", extra=["--role", "test-fixture"]),
        )
        # Typer 的 usage error 走 stderr（exit code 2），与 print_error 的 stdout 路径不同。
        output = _strip_ansi(result.stdout) + _strip_ansi(result.stderr)
        assert result.exit_code == 2
        assert "role must be one of" in output

    def test_unknown_profile_is_a_usage_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("LLMTRACE_TEST_KEY", _SECRET)
        result = runner.invoke(
            app,
            _capture_args(data_dir=tmp_path / "appdata", extra=["--profile", "turbo"]),
        )
        output = _strip_ansi(result.stdout) + _strip_ansi(result.stderr)
        assert result.exit_code == 2
        assert "profile must be one of" in output


# ---------------------------------------------------------------------------
# Secret scrubbing at the CLI boundary
# ---------------------------------------------------------------------------


class TestCaptureSecretScrubbing:
    def test_confirm_prompt_scrubs_credentials(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        captured_prompts: list[str] = []

        def capturing_confirm(message: str, *args: object, **kwargs: object) -> bool:
            captured_prompts.append(message)
            return False

        monkeypatch.setenv("LLMTRACE_TEST_KEY", _SECRET)
        monkeypatch.setattr("typer.confirm", capturing_confirm)
        # 伪 TTY stdin 让 isatty 门禁通过（typer.testing 的 isolation 硬编码了
        # io.BytesIO，故需替换其 make_input_stream）。
        monkeypatch.setattr("typer.testing.make_input_stream", _tty_bytes_io)

        # 去掉 --yes 才会走到 confirm 提示分支；confirm 返回 False 即用户拒绝。
        args = [arg for arg in _capture_args(data_dir=tmp_path / "appdata", base_url=_SECRET_URL) if arg != "--yes"]
        result = runner.invoke(app, args)
        stdout = _strip_ansi(result.stdout)
        all_text = stdout + " ".join(captured_prompts)

        assert len(captured_prompts) == 1, "confirm 提示必须被触发，否则本用例为空验证"
        assert "test.example.com" in captured_prompts[0]
        assert result.exit_code == 0
        assert not (tmp_path / "appdata" / "fingerprints").exists()
        for secret in ("myuser", "mypassword", "secret123"):
            assert secret not in all_text

    def test_exception_output_scrubs_secrets(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        async def failing_run(self: object, **kwargs: object) -> tuple[object, ...]:
            raise RuntimeError(f"capture failed: myuser:mypassword@evil.com api_key=secret123 {_SECRET}")

        monkeypatch.setenv("LLMTRACE_TEST_KEY", _SECRET)
        monkeypatch.setattr(FingerprintExecutor, "run", failing_run)

        with respx.mock(assert_all_called=False) as mock:
            mock_openai(mock, suite=load_fingerprint_suite())
            result = runner.invoke(
                app,
                _capture_args(data_dir=tmp_path / "appdata", base_url=_SECRET_URL),
            )

        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 1
        assert "fingerprint capture" in stdout
        for secret in ("myuser", "mypassword", "secret123", _SECRET):
            assert secret not in stdout


# ---------------------------------------------------------------------------
# Provider / budget wiring (Rule 4)
# ---------------------------------------------------------------------------


class TestBudgetWiring:
    def test_budget_covers_exactly_the_probe_requests(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """One request per (probe, repetition) and no more — the budget is exact."""
        suite = load_fingerprint_suite()
        repetitions = repetitions_for_profile(FingerprintProfile.QUICK)
        calls: list[httpx.Request] = []

        def responder(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(500, json={"error": "boom"})

        monkeypatch.setenv("LLMTRACE_TEST_KEY", _SECRET)
        with respx.mock(assert_all_called=False) as mock:
            mock.post(f"{BASE_URL}/chat/completions").mock(side_effect=responder)
            result = runner.invoke(app, _capture_args(data_dir=tmp_path / "appdata"))

        stdout = _strip_ansi(result.stdout)
        # HTTP 500 是"无效样本"而不是异常：分母不变，snapshot 仍然落盘（Task 8.4）。
        assert result.exit_code == 0, stdout
        assert len(calls) == len(suite.probes) * repetitions
        assert f"0 / {len(suite.probes) * repetitions}" in stdout
