"""CLI ``run --verify-model`` 集成测试（Task 22 / 35 / 36）.

覆盖三条用户路径：

* 默认 ``run`` 不变 —— 没有 ``--verify-model`` 时不出现任何指纹区块，请求上限仍是
  protocol + benchmark（``--no-streaming`` 下 4 + 32 = 36）；
* 身份证据是显式 opt-in 且 fail closed —— 没有兼容 reference set、多个兼容集合、
  轮数不兼容、``--fingerprint-set`` 缺少 ``--verify-model`` 都在发出任何请求前退出 1；
* 解析成功后 —— dry-run 展示 profile / set / 指纹请求数（36 + 6 × 8 = 84），真实执行
  写出三个指纹 artifact。

真实 HTTP 一律由 respx 拦截；执行用例把生产 sandbox 工厂替换为
``TrustedFakeBackend``，因此本文件不依赖 Docker，也不访问任何真实 endpoint。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest
import respx
from typer.main import get_command
from typer.testing import CliRunner

from llmtrace.cli import app
from llmtrace.execution.artifacts import RunArtifactRepository
from llmtrace.fingerprint.models import (
    FingerprintProfile,
    FingerprintSuite,
    repetitions_for_profile,
)
from llmtrace.fingerprint.suite import load_fingerprint_suite
from tests.execution.conftest import TrustedFakeBackend
from tests.fingerprint.conftest import (
    BASE_URL,
    MODEL_ID,
    FingerprintFixture,
    make_reference_snapshot,
    mock_openai,
    publish_reference_set,
    publish_validated_policy,
)

runner = CliRunner()
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
_API_KEY_ENV = "LLMTRACE_FP_RUN_KEY"
_API_KEY = "sk-fingerprint-run-secret"
#: ``--repeat 1 --no-streaming`` 下 protocol (4) + benchmark (32)；STANDARD 再加 6 × 8 = 48。
LEGACY_REQUESTS = 36
FINGERPRINT_REQUESTS = 48


_TABLE_BORDER = str.maketrans(dict.fromkeys("│┃|"))


@pytest.fixture
def suite() -> FingerprintSuite:
    """本次运行将执行的指纹套件（内置 categorical 套件）."""
    return load_fingerprint_suite()


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


def _compact(text: str) -> str:
    """去掉 ANSI、全部空白与表格分隔符：Rich 会按终端宽度折行，整段断言必须先归一化."""
    return "".join(_strip_ansi(text).split()).translate(_TABLE_BORDER)


def _fingerprint_set(
    data_root: Path,
    *,
    set_id: str = "cli-run-set",
    set_version: str = "1.0.0",
    snapshot_prefix: str = "cli-run",
) -> FingerprintFixture:
    """在 *data_root* 下发布一个两 identity 的 STANDARD reference set."""
    suite = load_fingerprint_suite()
    repetitions = repetitions_for_profile(FingerprintProfile.STANDARD)
    return publish_reference_set(
        data_root=data_root,
        suite=suite,
        set_id=set_id,
        set_version=set_version,
        snapshots=(
            make_reference_snapshot(
                suite=suite,
                snapshot_id=f"{snapshot_prefix}-claimed-capture-1",
                model_id=MODEL_ID,
                repetitions=repetitions,
            ),
            make_reference_snapshot(
                suite=suite,
                snapshot_id=f"{snapshot_prefix}-impostor-capture-1",
                model_id="other-model",
                repetitions=repetitions,
                choice_index=1,
            ),
        ),
    )


def _run_args(*, output_dir: Path, extra: list[str] | None = None) -> list[str]:
    """固定 ``--no-streaming`` 的 run 参数，使请求上限可精确断言."""
    args = [
        "run",
        "--protocol",
        "openai",
        "--base-url",
        BASE_URL,
        "--model",
        MODEL_ID,
        "--api-key-env",
        _API_KEY_ENV,
        "--output-dir",
        str(output_dir),
        "--repeat",
        "1",
        "--no-streaming",
    ]
    if extra:
        args.extend(extra)
    return args


#: The default answer the Quick Suite graders can both extract.
_GRADABLE_ANSWER = "The answer is (A). The answer is 42."


def _echoing_body(answer: str) -> dict[str, object]:
    """An OpenAI-compatible body that adversarially echoes the API key back."""
    return {
        "id": "chatcmpl-echo",
        "object": "chat.completion",
        "created": 1677652288,
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
        "echoed_secret": _API_KEY,
    }


class TestRunHelp:
    def test_run_help_lists_fingerprint_flags(self) -> None:
        """Task 35：run 暴露 --verify-model / --fingerprint-profile / --fingerprint-set."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0, _strip_ansi(result.stdout)
        # 帮助文本会被渲染宽度截断（``--fingerprint-pro…``），因此直接断言命令参数。
        options = {option for param in get_command(app).commands["run"].params for option in param.opts}
        assert {"--verify-model", "--fingerprint-profile", "--fingerprint-set"} <= options


class TestRunFingerprintGates:
    """身份证据输入有问题时，必须在任何请求之前 fail closed（exit 1）."""

    def test_verify_model_without_a_reference_set_exits_1(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("LLMTRACE_HOME", str(tmp_path / "appdata"))
        result = runner.invoke(
            app,
            _run_args(output_dir=tmp_path / "reports", extra=["--verify-model"]),
        )
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 1, stdout
        assert "未找到与本次运行兼容的 fingerprint reference set" in stdout

    def test_fingerprint_set_without_verify_model_exits_1(self, tmp_path: Path) -> None:
        """只给 --fingerprint-set 而不 opt-in → 明确拒绝，绝不静默忽略."""
        fixture = _fingerprint_set(tmp_path / "appdata")
        result = runner.invoke(
            app,
            _run_args(
                output_dir=tmp_path / "reports",
                extra=["--fingerprint-set", str(fixture.set_path)],
            ),
        )
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 1, stdout
        assert "--fingerprint-set 只在 --verify-model 下生效" in stdout

    def test_ambiguous_auto_discovery_exits_1(self, monkeypatch, tmp_path: Path) -> None:
        """Task 36：多个兼容集合绝不随机挑一个，列出候选并要求显式指定."""
        data_root = tmp_path / "appdata"
        first = _fingerprint_set(data_root, set_id="cli-run-set-a", snapshot_prefix="cli-run-a")
        second = _fingerprint_set(data_root, set_id="cli-run-set-b", snapshot_prefix="cli-run-b")
        monkeypatch.setenv("LLMTRACE_HOME", str(data_root))
        result = runner.invoke(
            app,
            _run_args(output_dir=tmp_path / "reports", extra=["--verify-model"]),
        )
        stdout = _strip_ansi(result.stdout)
        compact = _compact(stdout)
        assert result.exit_code == 1, stdout
        assert "Multiplecompatiblefingerprintreferencesetsfound" in compact
        assert first.reference_set.fingerprint_set_id in compact
        assert second.reference_set.fingerprint_set_id in compact

    def test_incompatible_repetitions_exit_1(self, monkeypatch, tmp_path: Path) -> None:
        """STANDARD reference set + QUICK profile → 分布不可比，发出请求前退出."""
        data_root = tmp_path / "appdata"
        fixture = _fingerprint_set(data_root)
        monkeypatch.setenv("LLMTRACE_HOME", str(data_root))
        result = runner.invoke(
            app,
            _run_args(
                output_dir=tmp_path / "reports",
                extra=[
                    "--verify-model",
                    "--fingerprint-profile",
                    "quick",
                    "--fingerprint-set",
                    str(fixture.set_path),
                ],
            ),
        )
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 1, stdout
        assert "distributions would not be comparable" in stdout

    def test_unknown_profile_exits_2(self, monkeypatch, tmp_path: Path) -> None:
        """未知成本档位在发出请求前被拒绝."""
        data_root = tmp_path / "appdata"
        fixture = _fingerprint_set(data_root)
        monkeypatch.setenv("LLMTRACE_HOME", str(data_root))
        result = runner.invoke(
            app,
            _run_args(
                output_dir=tmp_path / "reports",
                extra=["--verify-model", "--fingerprint-profile", "bogus", "--fingerprint-set", str(fixture.set_path)],
            ),
        )
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 2, stdout
        # typer.BadParameter 的 usage error 走 stderr；Rich 会截断错误面板中段，
        # 因此只断言保留下来的尾部（``got 'bogus'``）。
        assert "got'bogus'" in _compact(result.output)


class TestRunFingerprintDryRun:
    def test_default_run_has_no_fingerprint_block(self, monkeypatch, tmp_path: Path) -> None:
        """Task 45 / Rule 1：没有 --verify-model 的 run 保持旧语义."""
        monkeypatch.setenv("LLMTRACE_HOME", str(tmp_path / "appdata"))
        result = runner.invoke(app, _run_args(output_dir=tmp_path / "reports", extra=["--dry-run"]))
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 0, stdout
        assert "Dry Run" in stdout
        assert "Fingerprint" not in stdout
        assert f"总请求上限{LEGACY_REQUESTS}" in _compact(stdout)

    def test_unique_auto_discovery_enters_dry_run_with_fingerprint_requests(self, monkeypatch, tmp_path: Path) -> None:
        """Task 22 / 36：唯一兼容集合被自动采用，请求上限覆盖 6 × 8 指纹请求."""
        data_root = tmp_path / "appdata"
        fixture = _fingerprint_set(data_root)
        monkeypatch.setenv("LLMTRACE_HOME", str(data_root))
        result = runner.invoke(
            app,
            _run_args(output_dir=tmp_path / "reports", extra=["--verify-model", "--dry-run"]),
        )
        stdout = _strip_ansi(result.stdout)
        compact = _compact(stdout)
        assert result.exit_code == 0, stdout
        assert "FingerprintProfilestandard" in compact
        assert fixture.reference_set.fingerprint_set_id in stdout
        assert f"Fingerprint请求{FINGERPRINT_REQUESTS}" in compact
        assert f"总请求上限{LEGACY_REQUESTS + FINGERPRINT_REQUESTS}" in compact

    def test_without_a_policy_the_dry_run_says_unvalidated(self, monkeypatch, tmp_path: Path) -> None:
        """Task 43：只有 reference set、没有 decision policy → dry-run 标注 UNVALIDATED."""
        data_root = tmp_path / "appdata"
        fixture = _fingerprint_set(data_root)
        monkeypatch.setenv("LLMTRACE_HOME", str(data_root))
        result = runner.invoke(
            app,
            _run_args(
                output_dir=tmp_path / "reports",
                extra=["--verify-model", "--dry-run", "--fingerprint-set", str(fixture.set_path)],
            ),
        )
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 0, stdout
        assert "FingerprintPolicyUNVALIDATED" in _compact(stdout)

    def test_explicit_set_with_a_policy_shows_the_policy(self, monkeypatch, tmp_path: Path) -> None:
        data_root = tmp_path / "appdata"
        fixture = _fingerprint_set(data_root)
        policy = publish_validated_policy(fixture, policy_id="cli-run-policy", policy_version="2.0.0")
        monkeypatch.setenv("LLMTRACE_HOME", str(data_root))
        result = runner.invoke(
            app,
            _run_args(
                output_dir=tmp_path / "reports",
                extra=["--verify-model", "--dry-run", "--fingerprint-set", str(fixture.set_path)],
            ),
        )
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 0, stdout
        assert f"{policy.policy_id}v{policy.policy_version}" in _compact(stdout)


class TestRunFingerprintExecution:
    def test_verified_run_writes_the_fingerprint_artifacts(
        self, monkeypatch, tmp_path: Path, suite: FingerprintSuite
    ) -> None:
        """Task 35 / 55：``run --verify-model`` 全流程写出三个指纹 artifact 并计入预算."""
        data_root = tmp_path / "appdata"
        fixture = _fingerprint_set(data_root)
        policy = publish_validated_policy(fixture)
        output_dir = tmp_path / "reports"

        monkeypatch.setenv("LLMTRACE_HOME", str(data_root))
        monkeypatch.setenv(_API_KEY_ENV, _API_KEY)
        monkeypatch.setattr("llmtrace.execution.runner.create_code_execution_backend", TrustedFakeBackend)

        with respx.mock as mock:
            mock_openai(mock, suite=suite)
            result = runner.invoke(
                app,
                _run_args(
                    output_dir=output_dir,
                    extra=["--verify-model", "--yes", "--fingerprint-set", str(fixture.set_path)],
                ),
            )

        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 0, stdout
        # §Task 46：API key 绝不进入任何输出或产物。
        assert _API_KEY not in stdout

        run_dirs = [path for path in (output_dir / "runs").iterdir() if path.is_dir()]
        assert len(run_dirs) == 1
        artifacts = {path.name for path in run_dirs[0].iterdir()}
        assert {"fingerprint_snapshot.json", "fingerprint_match.json", "fingerprint_verification.json"} <= artifacts
        for name in ("fingerprint_snapshot.json", "fingerprint_match.json", "fingerprint_verification.json"):
            assert _API_KEY not in (run_dirs[0] / name).read_text(encoding="utf-8")

        manifest = RunArtifactRepository(output_dir).load_manifest(run_dirs[0].name)
        assert manifest.planned_requests == LEGACY_REQUESTS + FINGERPRINT_REQUESTS
        assert manifest.fingerprint_profile == FingerprintProfile.STANDARD.value
        assert manifest.fingerprint_set_id == fixture.reference_set.fingerprint_set_id

        # Task 42：控制台必须给出身份证据区块，并与免责声明同时出现。
        compact = _compact(stdout)
        assert "ModelVerification·Experimental" in compact
        assert f"ClaimedModel{MODEL_ID}" in compact
        assert f"FingerprintSet{fixture.reference_set.fingerprint_set_id}" in compact
        assert f"Policy{policy.policy_id}v{policy.policy_version}(validated)" in compact
        # 经过 held-out 验证的 policy + 命中声明模型的参考 → 允许 claim verdict。
        assert "ClaimConsistencyBEHAVIORCONSISTENTWITHCLAIM" in compact
        assert "FingerprintConfidenceHIGH" in compact
        assert "TopBehavioralMatches" in compact
        # Top-1 是行为最接近的参考（Rule 3），距离按 3 位小数渲染。
        assert f"1{MODEL_ID}distance0.000" in compact
        assert "Behavioralevidenceonly.Thisisnotcryptographicproofofupstreamidentity." in compact

    def test_unvalidated_run_ranks_but_prints_inconclusive(
        self, monkeypatch, tmp_path: Path, suite: FingerprintSuite
    ) -> None:
        """Task 43：只有 reference set、没有 validated policy → UNVALIDATED / INCONCLUSIVE."""
        data_root = tmp_path / "appdata"
        fixture = _fingerprint_set(data_root)
        output_dir = tmp_path / "reports"

        monkeypatch.setenv("LLMTRACE_HOME", str(data_root))
        monkeypatch.setenv(_API_KEY_ENV, _API_KEY)
        monkeypatch.setattr("llmtrace.execution.runner.create_code_execution_backend", TrustedFakeBackend)

        with respx.mock as mock:
            mock_openai(mock, suite=suite)
            result = runner.invoke(
                app,
                _run_args(
                    output_dir=output_dir,
                    extra=["--verify-model", "--yes", "--fingerprint-set", str(fixture.set_path)],
                ),
            )

        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 0, stdout
        assert "COMPLETED_WITH_WARNINGS" in stdout
        compact = _compact(stdout)
        # Rule 2 / Task 43：没有 validated policy 时绝不出现 claim verdict。
        assert "PolicyUNVALIDATED" in compact
        assert "ClaimConsistencyINCONCLUSIVE" in compact
        assert "BEHAVIORCONSISTENTWITHCLAIM" not in compact
        assert "BEHAVIORINCONSISTENTWITHCLAIM" not in compact
        # 但仍允许 Top-K 排序展示。
        assert "TopBehavioralMatches" in compact
        assert f"1{MODEL_ID}" in compact

    def test_echoed_secret_never_reaches_any_verify_model_artifact(
        self, monkeypatch, tmp_path: Path, suite: FingerprintSuite
    ) -> None:
        """Task 46 / Task 55：响应头与响应体回显的 key 不得进入任何产物.

        与默认 run 的 ``test_response_echo_never_persisted`` 等价，但这里额外启用
        ``--verify-model``：新增的 fingerprint 采集与 report 的 fingerprint section
        同样是持久化面，必须一起被清洗。
        """
        data_root = tmp_path / "appdata"
        fixture = _fingerprint_set(data_root)
        publish_validated_policy(fixture)
        output_dir = tmp_path / "reports"

        monkeypatch.setenv("LLMTRACE_HOME", str(data_root))
        monkeypatch.setenv(_API_KEY_ENV, _API_KEY)
        monkeypatch.setattr("llmtrace.execution.runner.create_code_execution_backend", TrustedFakeBackend)

        answers = {probe.prompt: probe.choices[0] for probe in suite.probes}

        def responder(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode("utf-8"))
            answer = _GRADABLE_ANSWER
            for message in payload.get("messages") or []:
                candidate = answers.get(message.get("content", ""))
                if candidate is not None:
                    answer = candidate
                    break
            return httpx.Response(
                200,
                json=_echoing_body(answer),
                headers={"content-type": "application/json", "x-debug": _API_KEY},
            )

        with respx.mock as mock:
            mock.get(f"{BASE_URL}/models").respond(
                status_code=200,
                json={"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]},
                headers={"x-debug": _API_KEY},
            )
            mock.post(f"{BASE_URL}/chat/completions").mock(side_effect=responder)
            result = runner.invoke(
                app,
                _run_args(
                    output_dir=output_dir,
                    extra=["--verify-model", "--yes", "--fingerprint-set", str(fixture.set_path)],
                ),
            )

        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 0, stdout
        assert _API_KEY not in stdout

        run_dirs = [path for path in (output_dir / "runs").iterdir() if path.is_dir()]
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]
        # 指纹阶段确实执行过，因此 report 的 fingerprint section 是被覆盖的持久化面之一。
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        assert "fingerprint" in report
        for artifact in run_dir.iterdir():
            assert _API_KEY not in artifact.read_text(encoding="utf-8"), f"leak in {artifact.name}"
