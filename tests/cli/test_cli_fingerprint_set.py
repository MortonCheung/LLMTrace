"""CLI ``fingerprint set-create`` / ``validate`` / ``inspect`` 测试（Task 40 / 41）.

覆盖：

* ``set-create`` 的成功装配，以及 Task 40 的 fail-closed 门禁 —— ``test_fixture``
  快照、被篡改的快照、不兼容的成员配置、重复 snapshot、重复 ``(set_id, set_version)``、
  不存在的 snapshot；
* ``validate`` 的 held-out 验证输出（Task 41 的八个字段）：数据不足时输出
  ``Validated NO`` 且 exit 0（不是异常），满足 Step 18.1 最低条件时输出
  ``Validated YES`` 并落盘 validated policy；
* ``inspect`` 的只读视图（仓库概况 / 单个 snapshot 的完整性与 ``--debug`` 堆栈）。

全部用例只写 ``tmp_path``，不发出任何 HTTP 请求。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from llmtrace.cli import app
from llmtrace.fingerprint.models import (
    FingerprintProfile,
    FingerprintSourceRole,
    FingerprintSuite,
    repetitions_for_profile,
)
from llmtrace.fingerprint.reference import FingerprintReferenceSnapshot, build_fingerprint_snapshot
from llmtrace.fingerprint.repository import FingerprintRepository
from llmtrace.fingerprint.suite import load_fingerprint_suite
from tests.fingerprint.conftest import (
    CAPTURED_AT,
    MODEL_ID,
    FingerprintFixture,
    make_observations,
    make_reference_snapshot,
    make_reference_snapshot_with_rounds,
    publish_reference_set,
    publish_validated_policy,
)

runner = CliRunner()

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    """剥离 ANSI 控制序列，返回纯文本."""
    return _ANSI_ESCAPE.sub("", text)


def _compact(text: str) -> str:
    """去掉 ANSI 与所有空白，便于跨 Rich 对齐/折行做子串断言."""
    return re.sub(r"\s+", "", _strip_ansi(text))


def _write_snapshots(data_dir: Path, snapshots: tuple[FingerprintReferenceSnapshot, ...]) -> FingerprintRepository:
    """把成员快照写进 ``data_dir`` 下的 fingerprint 仓库，返回重新加载的仓库."""
    repository = FingerprintRepository.load(data_root=data_dir)
    for snapshot in snapshots:
        repository.snapshots.save(snapshot)
    return FingerprintRepository.load(data_root=data_dir)


def _set_create_args(
    *,
    data_dir: Path,
    snapshot_ids: list[str],
    set_id: str = "cli-set",
    set_version: str = "0.1.0",
    extra: list[str] | None = None,
) -> list[str]:
    args = [
        "fingerprint",
        "set-create",
        "--set-id",
        set_id,
        "--set-version",
        set_version,
        "--data-dir",
        str(data_dir),
    ]
    for snapshot_id in snapshot_ids:
        args.extend(["--snapshot", snapshot_id])
    if extra is not None:
        args.extend(extra)
    return args


def _validate_args(
    *,
    data_dir: Path,
    set_id: str,
    set_version: str = "0.1.0",
    policy_id: str = "cli-policy",
    policy_version: str = "0.1.0",
    extra: list[str] | None = None,
) -> list[str]:
    args = [
        "fingerprint",
        "validate",
        "--set-id",
        set_id,
        "--set-version",
        set_version,
        "--policy-id",
        policy_id,
        "--policy-version",
        policy_version,
        "--data-dir",
        str(data_dir),
    ]
    if extra is not None:
        args.extend(extra)
    return args


def _two_signature_captures(suite: FingerprintSuite, *, repetitions: int) -> tuple[FingerprintReferenceSnapshot, ...]:
    """两个可区分 identity 的采集，供 set-create 的装配/门禁用例复用."""
    return (
        make_reference_snapshot(
            suite=suite,
            snapshot_id="baseline-a",
            model_id=MODEL_ID,
            repetitions=repetitions,
        ),
        make_reference_snapshot(
            suite=suite,
            snapshot_id="baseline-b",
            model_id="other-model",
            repetitions=repetitions,
            choice_index=1,
        ),
    )


def _publish_two_identity_set(data_dir: Path) -> FingerprintFixture:
    """两个 identity 的参考集（``inspect`` 展示与 ``validate`` 条件不足用例的输入）."""
    suite = load_fingerprint_suite()
    repetitions = repetitions_for_profile(FingerprintProfile.STANDARD)
    return publish_reference_set(
        data_root=data_dir,
        suite=suite,
        snapshots=(
            make_reference_snapshot(
                suite=suite,
                snapshot_id="ref-claimed-capture-1",
                model_id=MODEL_ID,
                repetitions=repetitions,
            ),
            make_reference_snapshot(
                suite=suite,
                snapshot_id="ref-impostor-capture-1",
                model_id="other-model",
                repetitions=repetitions,
                choice_index=1,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# set-create
# ---------------------------------------------------------------------------


class TestSetCreate:
    def test_set_create_assembles_and_saves_the_set(self, tmp_path: Path) -> None:
        suite = load_fingerprint_suite()
        repetitions = repetitions_for_profile(FingerprintProfile.STANDARD)
        data_dir = tmp_path / "appdata"
        _write_snapshots(data_dir, _two_signature_captures(suite, repetitions=repetitions))

        result = runner.invoke(
            app,
            _set_create_args(data_dir=data_dir, snapshot_ids=["baseline-a", "baseline-b"]),
        )

        stdout = _strip_ansi(result.stdout)
        compact = _compact(stdout)
        assert result.exit_code == 0, stdout
        assert "FingerprintSet complete" in stdout
        assert "Members:2captures" in compact
        assert "Identities:2" in compact
        assert "fingerprintvalidate" in compact

        repository = FingerprintRepository.load(data_root=data_dir)
        saved = repository.sets.get("cli-set", "0.1.0")
        repository.sets.verify("cli-set", "0.1.0")
        assert {member.snapshot_id for member in saved.members} == {"baseline-a", "baseline-b"}
        assert saved.repetitions == repetitions

    def test_set_create_rejects_test_fixture_snapshots(self, tmp_path: Path) -> None:
        """Step 14.2：production 路径没有 ``--allow-test-fixture``，fixture 进不了 trusted set."""
        suite = load_fingerprint_suite()
        repetitions = repetitions_for_profile(FingerprintProfile.STANDARD)
        data_dir = tmp_path / "appdata"
        _write_snapshots(
            data_dir,
            (
                build_fingerprint_snapshot(
                    snapshot_id="fixture-capture",
                    model_id=MODEL_ID,
                    provider_id="openai",
                    source_role=FingerprintSourceRole.TEST_FIXTURE,
                    suite=suite,
                    repetitions=repetitions,
                    observations=make_observations(suite, repetitions),
                    captured_at=CAPTURED_AT,
                ),
            ),
        )

        result = runner.invoke(app, _set_create_args(data_dir=data_dir, snapshot_ids=["fixture-capture"]))

        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 1
        assert "test_fixture" in stdout
        assert len(FingerprintRepository.load(data_root=data_dir).sets.list()) == 0

    def test_set_create_rejects_a_tampered_snapshot(self, tmp_path: Path) -> None:
        """磁盘字节被改过、内容身份对不上时 fail closed（不静默接受）。"""
        suite = load_fingerprint_suite()
        repetitions = repetitions_for_profile(FingerprintProfile.STANDARD)
        data_dir = tmp_path / "appdata"
        _write_snapshots(
            data_dir,
            (
                make_reference_snapshot(
                    suite=suite, snapshot_id="baseline-a", model_id=MODEL_ID, repetitions=repetitions
                ),
            ),
        )

        path = data_dir / "fingerprints" / "snapshots" / "baseline-a.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["model_id"] = "tampered-model"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        result = runner.invoke(app, _set_create_args(data_dir=data_dir, snapshot_ids=["baseline-a"]))

        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 1
        assert "fingerprint set-create" in stdout
        assert "content hash mismatch" in stdout
        assert len(FingerprintRepository.load(data_root=data_dir).sets.list()) == 0

    def test_set_create_rejects_incompatible_repetitions(self, tmp_path: Path) -> None:
        suite = load_fingerprint_suite()
        data_dir = tmp_path / "appdata"
        _write_snapshots(
            data_dir,
            (
                make_reference_snapshot(
                    suite=suite,
                    snapshot_id="baseline-standard",
                    model_id=MODEL_ID,
                    repetitions=repetitions_for_profile(FingerprintProfile.STANDARD),
                ),
                make_reference_snapshot(
                    suite=suite,
                    snapshot_id="baseline-quick",
                    model_id="other-model",
                    repetitions=repetitions_for_profile(FingerprintProfile.QUICK),
                    choice_index=1,
                ),
            ),
        )

        result = runner.invoke(
            app,
            _set_create_args(data_dir=data_dir, snapshot_ids=["baseline-standard", "baseline-quick"]),
        )

        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 1
        assert "incompatible" in stdout
        assert "repetitions" in stdout

    def test_set_create_rejects_duplicate_snapshot_members(self, tmp_path: Path) -> None:
        suite = load_fingerprint_suite()
        repetitions = repetitions_for_profile(FingerprintProfile.STANDARD)
        data_dir = tmp_path / "appdata"
        _write_snapshots(
            data_dir,
            (
                make_reference_snapshot(
                    suite=suite, snapshot_id="baseline-a", model_id=MODEL_ID, repetitions=repetitions
                ),
            ),
        )

        result = runner.invoke(
            app,
            _set_create_args(data_dir=data_dir, snapshot_ids=["baseline-a", "baseline-a"]),
        )

        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 1
        assert "duplicate snapshot_id" in stdout

    def test_set_create_rejects_unknown_snapshot(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            _set_create_args(data_dir=tmp_path / "appdata", snapshot_ids=["ghost"]),
        )
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 1
        assert "fingerprint set-create" in stdout
        assert "not found" in stdout

    def test_set_create_rejects_a_duplicate_set_revision(self, tmp_path: Path) -> None:
        """集合 append-only：同 ``(set_id, set_version)`` 第二次必须被拒。"""
        suite = load_fingerprint_suite()
        repetitions = repetitions_for_profile(FingerprintProfile.STANDARD)
        data_dir = tmp_path / "appdata"
        _write_snapshots(data_dir, _two_signature_captures(suite, repetitions=repetitions))

        args = _set_create_args(data_dir=data_dir, snapshot_ids=["baseline-a", "baseline-b"])
        first = runner.invoke(app, args)
        second = runner.invoke(app, args)

        assert first.exit_code == 0, _strip_ansi(first.stdout)
        stdout = _strip_ansi(second.stdout)
        assert second.exit_code == 1
        assert "fingerprint set-create" in stdout
        assert "already exists" in stdout

    def test_set_create_debug_flag_prints_traceback(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            _set_create_args(data_dir=tmp_path / "appdata", snapshot_ids=["ghost"], extra=["--debug"]),
        )
        output = _strip_ansi(result.stdout) + _strip_ansi(result.stderr)
        assert result.exit_code == 1
        assert "Traceback (most recent call last)" in output


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def _five_identity_snapshots(repetitions: int) -> tuple[FingerprintReferenceSnapshot, ...]:
    """满足 Step 18.1 的参考集：5 个 identity × 每个 2 次独立 capture.

    前四个 identity 各自固定答一个 choice；第五个在两个 choice 之间对半混答，
    因此与四个 one-hot 分布的距离都大于 0（否则会被自己的 impostor 淹没，
    任何 threshold 都无法满足 FAR 约束）。
    """
    suite = load_fingerprint_suite()
    snapshots: list[FingerprintReferenceSnapshot] = []
    for index in range(4):
        for capture in range(2):
            snapshots.append(
                make_reference_snapshot(
                    suite=suite,
                    snapshot_id=f"ref-{index}-capture-{capture}",
                    model_id=f"model-{index}",
                    repetitions=repetitions,
                    choice_index=index,
                )
            )
    mixed_rounds = [0, 1] * (repetitions // 2)
    for capture in range(2):
        snapshots.append(
            make_reference_snapshot_with_rounds(
                suite=suite,
                snapshot_id=f"ref-4-capture-{capture}",
                model_id="model-4",
                choice_index_by_round=mixed_rounds,
            )
        )
    return tuple(snapshots)


class TestValidate:
    def test_validate_reports_insufficient_data_without_failing(self, tmp_path: Path) -> None:
        """Step 18.1：条件不足时 ``Validated NO`` 是结果而不是异常（exit 0、不发布数字）."""
        data_dir = tmp_path / "appdata"
        _publish_two_identity_set(data_dir)
        result = runner.invoke(
            app,
            _validate_args(
                data_dir=data_dir,
                set_id="test-identity-set",
                extra=["--min-comparable-probes", "6"],
            ),
        )

        stdout = _strip_ansi(result.stdout)
        compact = _compact(stdout)
        assert result.exit_code == 0, stdout
        assert "Fingerprint validation complete" in stdout
        assert "Identitycount:2" in compact
        assert "Held-outcaptures:0" in compact
        assert "Top-1:n/a" in compact
        assert "Top-3:n/a" in compact
        assert "TPR:n/a" in compact
        assert "FAR:n/a" in compact
        assert "Threshold:n/a" in compact
        assert "ValidatedNO" in compact
        assert "Reason:" in stdout

        repository = FingerprintRepository.load(data_root=data_dir)
        policy = repository.policies.get("cli-policy", "0.1.0")
        assert policy.validated is False
        assert policy.distance_threshold is None

    def test_validate_publishes_a_validated_policy(self, tmp_path: Path) -> None:
        suite = load_fingerprint_suite()
        repetitions = repetitions_for_profile(FingerprintProfile.STANDARD)
        data_dir = tmp_path / "appdata"
        publish_reference_set(
            data_root=data_dir,
            suite=suite,
            snapshots=_five_identity_snapshots(repetitions),
            set_id="five-identity-set",
            set_version="1.0.0",
        )

        result = runner.invoke(
            app,
            _validate_args(data_dir=data_dir, set_id="five-identity-set", set_version="1.0.0"),
        )

        stdout = _strip_ansi(result.stdout)
        compact = _compact(stdout)
        assert result.exit_code == 0, stdout
        assert "Identitycount:5" in compact
        assert "Held-outcaptures:10" in compact
        assert "Top-1:1.00" in compact
        assert "Top-3:1.00" in compact
        assert "TPR:1.00" in compact
        assert "FAR:0.00" in compact
        assert "Threshold:0.0000" in compact
        assert "ValidatedYES" in compact
        assert "Reason:" not in stdout

        repository = FingerprintRepository.load(data_root=data_dir)
        repository.policies.verify("cli-policy", "0.1.0")
        policy = repository.policies.get("cli-policy", "0.1.0")
        assert policy.validated is True
        assert policy.distance_threshold == 0.0
        assert policy.fingerprint_set_id == "five-identity-set"

    def test_validate_unknown_set_exits_1(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            _validate_args(data_dir=tmp_path / "appdata", set_id="ghost"),
        )
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 1
        assert "fingerprint validate" in stdout
        assert "not found" in stdout

    def test_validate_debug_flag_prints_traceback(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            _validate_args(data_dir=tmp_path / "appdata", set_id="ghost", extra=["--debug"]),
        )
        output = _strip_ansi(result.stdout) + _strip_ansi(result.stderr)
        assert result.exit_code == 1
        assert "Traceback (most recent call last)" in output


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


class TestInspect:
    def test_inspect_lists_repository_contents(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "appdata"
        fixture = _publish_two_identity_set(data_dir)
        publish_validated_policy(fixture, policy_id="cli-policy", policy_version="0.1.0")
        result = runner.invoke(app, ["fingerprint", "inspect", "--data-dir", str(data_dir)])

        stdout = _strip_ansi(result.stdout)
        compact = _compact(stdout)
        assert result.exit_code == 0, stdout
        assert "Fingerprint repository" in stdout
        assert "Snapshots:2" in compact
        assert "Sets:1" in compact
        assert "Policies:1" in compact
        assert "ref-claimed-capture-1" in stdout
        assert "test-identity-setv0.1.0" in compact
        assert "cli-policyv0.1.0validated=YES" in compact

    def test_inspect_single_snapshot_reports_integrity(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "appdata"
        _publish_two_identity_set(data_dir)
        result = runner.invoke(
            app,
            ["fingerprint", "inspect", "--snapshot", "ref-claimed-capture-1", "--data-dir", str(data_dir)],
        )

        stdout = _strip_ansi(result.stdout)
        compact = _compact(stdout)
        assert result.exit_code == 0, stdout
        assert "Snapshot:ref-claimed-capture-1" in compact
        assert f"Identity:openai/{MODEL_ID}" in compact
        assert "Role:operator-assertedtrustedreference" in compact
        assert "Rounds:8" in compact
        assert "Samples:48/48valid" in compact
        assert "Integrity:verified(" in compact
        # 长路径会被 Rich 折行，故在去空白后的文本上断言文件名。
        assert "snapshots/ref-claimed-capture-1.json" in compact

    def test_inspect_unknown_snapshot_exits_1(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "appdata"
        _publish_two_identity_set(data_dir)
        result = runner.invoke(
            app,
            ["fingerprint", "inspect", "--snapshot", "ghost", "--data-dir", str(data_dir)],
        )
        stdout = _strip_ansi(result.stdout)
        assert result.exit_code == 1
        assert "fingerprint inspect" in stdout
        assert "not found" in stdout

    def test_inspect_debug_flag_prints_traceback(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "appdata"
        _publish_two_identity_set(data_dir)
        result = runner.invoke(
            app,
            [
                "fingerprint",
                "inspect",
                "--snapshot",
                "ghost",
                "--data-dir",
                str(data_dir),
                "--debug",
            ],
        )
        output = _strip_ansi(result.stdout) + _strip_ansi(result.stderr)
        assert result.exit_code == 1
        assert "Traceback (most recent call last)" in output
