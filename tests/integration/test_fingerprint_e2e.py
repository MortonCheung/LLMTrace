"""Task 56：身份证据端到端四场景（A 装配+验证 / B 候选一致 / C 未知候选 / D 混合切换）.

本文件的"端到端"具体指两条真实链路：

1. **参考数据的产生**（Scenario A）—— 10 次真实的 ``fingerprint capture`` CLI 调用
   （每次请求都经由 Provider → RequestBudget → EvidenceRecorder，Rule 4），随后
   ``set-create`` 与 ``validate`` 从同一个磁盘仓库装配参考集并发布 validated policy；
2. **候选的验证**（Scenario B / C / D）—— 真实 ``UnifiedAuditRunner``：预检解析同一份
   CLI 产物 → 指纹采集 → 匹配 → Rule 2 门禁 → report.json / report.html 落盘。

全部 HTTP 由 respx 提供 deterministic 的 OpenAI-compatible mock（Task 49），没有任何
真实端点调用；所有写入都落在 ``tmp_path_factory`` / ``tmp_path`` 下。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
import respx
from typer.testing import CliRunner

from llmtrace.analysis.routing import RoutingStabilityLevel
from llmtrace.cli import app
from llmtrace.config import AuditConfig, Protocol
from llmtrace.execution.artifacts import RunArtifactRepository
from llmtrace.execution.models import UnifiedRunStatus
from llmtrace.execution.runner import UnifiedAuditRunner
from llmtrace.fingerprint.models import (
    FingerprintMatchStatus,
    FingerprintProfile,
    repetitions_for_profile,
)
from llmtrace.fingerprint.repository import FingerprintRepository
from llmtrace.fingerprint.routing import SIGNAL_MULTIPLE_RESPONSE_MODELS, RoutingEvidenceStrength
from llmtrace.fingerprint.suite import load_fingerprint_suite
from tests.fingerprint.conftest import (
    API_KEY,
    BASE_URL,
    MOCK_MODEL_A,
    MOCK_MODEL_B,
    make_runner,
    mock_fingerprint_model,
)

SUITE = load_fingerprint_suite()
PROBE_COUNT = len(SUITE.probes)
STANDARD_REPETITIONS = repetitions_for_profile(FingerprintProfile.STANDARD)
FINGERPRINT_REQUESTS = PROBE_COUNT * STANDARD_REPETITIONS
#: Protocol (4) + Quick Suite (32)；指纹请求数在其之上叠加（Rule 1：增量）。
LEGACY_REQUESTS = 36

SET_ID = "official-mock-identities"
SET_VERSION = "1.0.0"
POLICY_ID = "official-mock-policy"
POLICY_VERSION = "1.0.0"

CAPTURE_KEY_ENV = "LLMTRACE_E2E_KEY"

#: Scenario C 的被审计端点：声称的 model label 不在参考集里。
UNKNOWN_MODEL = "mock-identity-unknown"
#: Scenario A 的第三 / 第四 / 第五个参考 identity。
MOCK_MODEL_C = "mock-identity-c"
MOCK_MODEL_D = "mock-identity-d"
MOCK_MODEL_MIXED = "mock-identity-mixed"

#: Scenario A 的五种 identity：四个 one-hot 行为 + 一个在两个 choice 之间切换的行为。
#: 第五个必须与四个 one-hot 都保持 > 0 的距离，否则集合无法满足 FAR 约束（Step 18.1）。
_IDENTITIES: tuple[tuple[str, tuple[int, ...]], ...] = (
    (MOCK_MODEL_A, (0,)),
    (MOCK_MODEL_B, (1,)),
    (MOCK_MODEL_C, (2,)),
    (MOCK_MODEL_D, (3,)),
    (MOCK_MODEL_MIXED, (0, 1) * (STANDARD_REPETITIONS // 2)),
)
_CAPTURES_PER_IDENTITY = 2

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    """剥离 ANSI 控制序列，返回纯文本."""
    return _ANSI_ESCAPE.sub("", text)


def _compact(text: str) -> str:
    """去掉所有空白，便于跨 Rich 对齐/折行做子串断言."""
    return re.sub(r"\s+", "", text)


@dataclass(frozen=True)
class OfficialReference:
    """Scenario A 的真实 CLI 产物：capture × 10 → set → validated policy."""

    data_root: Path
    set_path: Path
    repository: FingerprintRepository
    capture_stdout: tuple[str, ...]
    validate_stdout: str


def _snapshot_id(model_id: str, capture_index: int) -> str:
    return f"{model_id}-capture-{capture_index}"


def _capture_args(*, data_dir: Path, model: str, snapshot_id: str) -> list[str]:
    return [
        "fingerprint",
        "capture",
        "--protocol",
        "openai",
        "--base-url",
        BASE_URL,
        "--model",
        model,
        "--role",
        "official-baseline",
        "--profile",
        "standard",
        "--api-key-env",
        CAPTURE_KEY_ENV,
        "--snapshot-id",
        snapshot_id,
        "--data-dir",
        str(data_dir),
        "--yes",
    ]


def _set_create_args(*, data_dir: Path) -> list[str]:
    args = [
        "fingerprint",
        "set-create",
        "--set-id",
        SET_ID,
        "--set-version",
        SET_VERSION,
        "--data-dir",
        str(data_dir),
    ]
    for model_id, _ in _IDENTITIES:
        for capture_index in range(1, _CAPTURES_PER_IDENTITY + 1):
            args.extend(["--snapshot", _snapshot_id(model_id, capture_index)])
    return args


def _validate_args(*, data_dir: Path) -> list[str]:
    return [
        "fingerprint",
        "validate",
        "--set-id",
        SET_ID,
        "--set-version",
        SET_VERSION,
        "--policy-id",
        POLICY_ID,
        "--policy-version",
        POLICY_VERSION,
        "--data-dir",
        str(data_dir),
    ]


def _audit_config(model: str) -> AuditConfig:
    return AuditConfig(
        protocol=Protocol.OPENAI,
        base_url=BASE_URL,
        model=model,
        api_key_env="TEST_KEY",
        repeat_count=1,
        max_output_tokens=64,
        check_streaming=False,
        output_dir="reports",
    )


def _fingerprint_runner(
    model: str,
    repository: RunArtifactRepository,
    reference: OfficialReference,
) -> UnifiedAuditRunner:
    """对 CLI 产出的参考集启用身份证据的 runner（预检解析同一份磁盘数据）."""
    return make_runner(
        _audit_config(model),
        repository,
        verify_model=True,
        fingerprint_profile=FingerprintProfile.STANDARD,
        fingerprint_set_path=reference.set_path,
        fingerprint_repository=reference.repository,
    )


def _report_of(repository_root: Path, execution_id: str) -> dict[str, object]:
    path = repository_root / "runs" / execution_id / "report.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", API_KEY)


@pytest.fixture(scope="module")
def official_reference(tmp_path_factory: pytest.TempPathFactory) -> OfficialReference:
    """Scenario A：用真实 CLI 采集 5 个 identity × 2 次，装配参考集并做 held-out 验证."""
    data_root = tmp_path_factory.mktemp("fingerprint-home")
    cli = CliRunner()
    capture_stdout: list[str] = []

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(CAPTURE_KEY_ENV, API_KEY)
        for model_id, rounds in _IDENTITIES:
            for capture_index in range(1, _CAPTURES_PER_IDENTITY + 1):
                with respx.mock(assert_all_called=False) as mock:
                    mock_fingerprint_model(mock, suite=SUITE, choice_index_by_round=rounds)
                    captured = cli.invoke(
                        app,
                        _capture_args(
                            data_dir=data_root,
                            model=model_id,
                            snapshot_id=_snapshot_id(model_id, capture_index),
                        ),
                    )
                stdout = _strip_ansi(captured.stdout)
                assert captured.exit_code == 0, stdout
                capture_stdout.append(stdout)

        created = cli.invoke(app, _set_create_args(data_dir=data_root))
        assert created.exit_code == 0, _strip_ansi(created.stdout)

        validated = cli.invoke(app, _validate_args(data_dir=data_root))
        validate_stdout = _strip_ansi(validated.stdout)
        assert validated.exit_code == 0, validate_stdout

    repository = FingerprintRepository.load(data_root=data_root)
    return OfficialReference(
        data_root=data_root,
        set_path=repository.layout.fingerprint_sets_dir / f"{SET_ID}_{SET_VERSION}.json",
        repository=repository,
        capture_stdout=tuple(capture_stdout),
        validate_stdout=validate_stdout,
    )


class TestScenarioAReferenceSetFromRealCaptures:
    """Scenario A：>= 5 identities → set → validate（全部走 CLI，0 真实端点）."""

    def test_captures_build_a_reference_set_that_passes_held_out_validation(
        self, official_reference: OfficialReference
    ) -> None:
        reference = official_reference
        expected_captures = len(_IDENTITIES) * _CAPTURES_PER_IDENTITY
        assert len(reference.capture_stdout) == expected_captures == 10
        assert all("Fingerprint capture complete" in stdout for stdout in reference.capture_stdout)
        # 每次采集都在一个 provider 生命周期内跑满 6 probes × 8 repetitions。
        valid_samples = f"Samples{FINGERPRINT_REQUESTS}/{FINGERPRINT_REQUESTS}"
        assert all(valid_samples in _compact(stdout) for stdout in reference.capture_stdout)

        compact = _compact(reference.validate_stdout)
        assert "Fingerprint validation complete" in reference.validate_stdout
        assert "Identitycount:5" in compact
        assert "Held-outcaptures:10" in compact
        assert "Top-1:1.00" in compact
        assert "Top-3:1.00" in compact
        assert "TPR:1.00" in compact
        assert "FAR:0.00" in compact
        assert "Threshold:0.0000" in compact
        assert "ValidatedYES" in compact

        # 磁盘上的 append-only 仓库确实持有 10 个 snapshot 与 1 个集合。
        assert len(reference.repository.snapshots.list()) == expected_captures
        assert reference.repository.sets.get(SET_ID, SET_VERSION).repetitions == STANDARD_REPETITIONS
        assert reference.set_path.is_file()

        # 只有 Validated YES 的 policy 才带 production threshold（Step 18.1 / Rule 2）。
        reference.repository.policies.verify(POLICY_ID, POLICY_VERSION)
        policy = reference.repository.policies.get(POLICY_ID, POLICY_VERSION)
        assert policy.validated is True
        assert policy.distance_threshold == 0.0
        assert policy.fingerprint_set_id == SET_ID
        assert policy.minimum_comparable_probes == PROBE_COUNT


class TestScenarioBCandidateMatchesItsClaim:
    """Scenario B：Candidate A → verify → A top-ranked → consistent（policy validated）."""

    @pytest.mark.asyncio
    async def test_verified_candidate_is_ranked_first_under_its_own_claim(
        self, official_reference: OfficialReference, tmp_path: Path
    ) -> None:
        repository = RunArtifactRepository(tmp_path)
        runner = _fingerprint_runner(MOCK_MODEL_A, repository, official_reference)

        with respx.mock as mock:
            mock_fingerprint_model(
                mock,
                suite=SUITE,
                choice_index_by_round=(0,),
                response_model=MOCK_MODEL_A,
            )
            result = await runner.run()

        assert result.status is UnifiedRunStatus.COMPLETED
        match = result.fingerprint_match
        assert match is not None
        assert match.status is FingerprintMatchStatus.CONSISTENT_WITH_CLAIM
        assert match.entries[0].model_id == MOCK_MODEL_A
        assert match.claimed_model_id == MOCK_MODEL_A
        assert match.claimed_reference_distance == pytest.approx(0.0)

        verification = result.fingerprint_verification
        assert verification is not None
        assert verification.claim_verdict_produced is True
        assert verification.policy is not None
        assert verification.policy.validated is True
        assert verification.policy.policy_id == POLICY_ID

        # 预算与 provenance 覆盖 protocol + benchmark + fingerprint（增量，不改变前两者）。
        manifest = repository.load_manifest(result.execution_id)
        assert manifest.planned_requests == LEGACY_REQUESTS + FINGERPRINT_REQUESTS
        assert manifest.actual_requests == LEGACY_REQUESTS + FINGERPRINT_REQUESTS
        assert manifest.fingerprint_profile == FingerprintProfile.STANDARD.value
        assert manifest.fingerprint_set_id == SET_ID
        assert manifest.fingerprint_set_version == SET_VERSION

        section = _report_of(tmp_path, result.execution_id)["fingerprint"]
        assert section["reference_set"]["fingerprint_set_id"] == SET_ID
        assert section["decision_policy"]["policy_id"] == POLICY_ID
        assert section["decision_policy"]["validated"] is True
        assert section["verdict"]["match_status"] == FingerprintMatchStatus.CONSISTENT_WITH_CLAIM.value
        assert section["verdict"]["claim_verdict_produced"] is True
        assert section["jsd_reference"]["basis"] == "claimed_reference"
        assert section["aggregate_jsd"] == pytest.approx(0.0)
        assert section["top_k"][0]["model_id"] == MOCK_MODEL_A
        assert section["probe_coverage"]["compared_probes"] == PROBE_COUNT
        assert section["invalid_outcome_count"] == 0

        html = (tmp_path / "runs" / result.execution_id / "report.html").read_text(encoding="utf-8")
        assert "Model Fingerprint Evidence (Experimental)" in html


class TestScenarioCUnknownCandidateIsNeverAssignedAnIdentity:
    """Scenario C：Candidate unknown → 不给身份结论（即使 policy 已验证）."""

    @pytest.mark.asyncio
    async def test_unknown_claim_keeps_the_ranking_as_evidence_only(
        self, official_reference: OfficialReference, tmp_path: Path
    ) -> None:
        repository = RunArtifactRepository(tmp_path)
        runner = _fingerprint_runner(UNKNOWN_MODEL, repository, official_reference)

        with respx.mock as mock:
            mock_fingerprint_model(
                mock,
                suite=SUITE,
                choice_index_by_round=(0,),
                response_model=UNKNOWN_MODEL,
            )
            result = await runner.run()

        match = result.fingerprint_match
        assert match is not None
        # 行为与参考 A 完全一致，但 A 只是"最接近的参考"，不是这个未知端点的身份（Rule 3）。
        assert match.entries[0].model_id == MOCK_MODEL_A
        assert all(entry.model_id != UNKNOWN_MODEL for entry in match.entries)
        # claimed label 没有参考条目 → 没有可比较的距离 → 只能是排序，不能是判定。
        assert match.claimed_model_id == UNKNOWN_MODEL
        assert match.claimed_reference_distance is None
        assert match.status is FingerprintMatchStatus.RANKED_ONLY

        verification = result.fingerprint_verification
        assert verification is not None
        # 门禁不是"没有 policy"：policy 存在且已验证，仍然不给结论（Rule 2）。
        assert verification.policy is not None
        assert verification.policy.validated is True
        assert verification.claim_verdict_produced is False

        section = _report_of(tmp_path, result.execution_id)["fingerprint"]
        assert section["verdict"]["match_status"] == FingerprintMatchStatus.RANKED_ONLY.value
        assert section["verdict"]["claim_verdict_produced"] is False
        assert section["claimed_model_id"] == UNKNOWN_MODEL
        assert section["claimed_reference_distance"] is None
        assert section["top_k"][0]["model_id"] == MOCK_MODEL_A
        assert section["disclaimer"]


class TestScenarioDSwitchingMockRaisesRoutingEvidence:
    """Scenario D：mixed / switching mock → routing 报 Suspicious."""

    @pytest.mark.asyncio
    async def test_alternating_response_model_is_reported_as_suspicious_routing(
        self, official_reference: OfficialReference, tmp_path: Path
    ) -> None:
        repository = RunArtifactRepository(tmp_path)
        runner = _fingerprint_runner(MOCK_MODEL_A, repository, official_reference)

        with respx.mock as mock:
            mock_fingerprint_model(
                mock,
                suite=SUITE,
                choice_index_by_round=(0,),
                response_model_by_request=(MOCK_MODEL_A, MOCK_MODEL_B),
            )
            result = await runner.run()

        # 身份证据是增量功能（Rule 1）：routing 报 Suspicious 只进入 report 的
        # fingerprint section，不改变既有 capability/calibration 的 run status。
        assert result.status is UnifiedRunStatus.COMPLETED
        assert result.fingerprint_snapshot is not None

        routing = _report_of(tmp_path, result.execution_id)["fingerprint"]["routing"]
        assert routing["level"] == RoutingStabilityLevel.SUSPICIOUS.value
        assert routing["distinct_response_models"] == 2
        assert routing["dominant_model_ratio"] is not None
        assert routing["dominant_model_ratio"] < 0.9
        split_signal = next(
            signal for signal in routing["signals"] if signal["signal_id"] == SIGNAL_MULTIPLE_RESPONSE_MODELS
        )
        assert split_signal["observed"] is True
        assert split_signal["strength"] == RoutingEvidenceStrength.STRONG.value
        # 措辞只描述行为，不给伪造比例或身份归属（Task 31 / Rule 7）。
        assert any("routing inconsistency observed" in reason for reason in routing["reasons"])
