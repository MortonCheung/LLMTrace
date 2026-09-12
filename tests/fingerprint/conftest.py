"""Shared fixtures for the v0.6 fingerprint domain and runner-integration tests.

Everything here is *fixture* data: deterministic observations, a published
on-disk reference set, and an OpenAI-compatible mock that answers each
fingerprint probe with a fixed choice.  No test in this package calls a real
endpoint, and no test writes outside ``tmp_path``.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx

from llmtrace.config import AuditConfig, Protocol
from llmtrace.execution.artifacts import RunArtifactRepository
from llmtrace.execution.runner import UnifiedAuditRunner
from llmtrace.fingerprint.models import (
    INVALID_OUTCOME,
    FingerprintProfile,
    FingerprintSampleObservation,
    FingerprintSourceRole,
    FingerprintSuite,
    repetitions_for_profile,
)
from llmtrace.fingerprint.policy import FingerprintDecisionPolicy, build_fingerprint_policy
from llmtrace.fingerprint.reference import FingerprintReferenceSnapshot, build_fingerprint_snapshot
from llmtrace.fingerprint.reference_set import FingerprintReferenceSet, FingerprintReferenceSetBuilder
from llmtrace.fingerprint.repository import FingerprintRepository
from llmtrace.fingerprint.suite import load_fingerprint_suite
from tests.execution.conftest import TrustedFakeBackend

API_KEY = "sk-super-secret-123"
TARGET_ID = "openai-test-target"
BASE_URL = "http://test.example.com/v1"
MODEL_ID = "my-real-model"

#: An answer the Quick Suite's ARC / GSM8K graders can both extract.
GRADABLE_ANSWER = "The answer is (A). The answer is 42."

#: Non-empty output the exact-choice normalizer cannot map to a choice (Task 49).
INVALID_MOCK_ANSWER = "I will not choose."

#: The two deterministic mock identities Task 56's scenarios replay.
MOCK_MODEL_A = "mock-identity-a"
MOCK_MODEL_B = "mock-identity-b"

#: A stable capture timestamp keeps fixture content hashes reproducible.
CAPTURED_AT = datetime(2026, 8, 1, tzinfo=UTC)


@pytest.fixture
def config() -> AuditConfig:
    """Minimal OpenAI-protocol audit config (one repetition, no streaming)."""
    return AuditConfig(
        protocol=Protocol.OPENAI,
        base_url=BASE_URL,
        model=MODEL_ID,
        api_key_env="TEST_KEY",
        repeat_count=1,
        max_output_tokens=64,
        check_streaming=False,
        output_dir="reports",
    )


@pytest.fixture
def api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", API_KEY)


@pytest.fixture
def suite() -> FingerprintSuite:
    """The built-in fingerprint suite that every run executes."""
    return load_fingerprint_suite()


# ---------------------------------------------------------------------------
# Deterministic observations / snapshots
# ---------------------------------------------------------------------------


def make_observations(
    suite: FingerprintSuite,
    repetitions: int,
    *,
    choice_index: int = 0,
    valid: bool = True,
    ref_prefix: str = "fixture",
) -> tuple[FingerprintSampleObservation, ...]:
    """One deterministic observation per (probe, round).

    ``valid=True`` always answers ``choices[choice_index]``; ``valid=False``
    always lands on the reserved invalid outcome — the "no request is ever
    dropped" shape the denominator depends on.
    """
    observations: list[FingerprintSampleObservation] = []
    for round_index in range(repetitions):
        for sequence_index, probe in enumerate(suite.probes):
            observations.append(
                FingerprintSampleObservation(
                    probe_id=probe.probe_id,
                    sequence_index=sequence_index,
                    round_index=round_index,
                    outcome=probe.choices[choice_index] if valid else INVALID_OUTCOME,
                    valid=valid,
                    response_body_sha256="a" * 64,
                    evidence_ref=f"{ref_prefix}:{probe.probe_id}:{round_index}",
                )
            )
    return tuple(observations)


def make_reference_snapshot(
    *,
    suite: FingerprintSuite,
    snapshot_id: str,
    model_id: str,
    repetitions: int,
    choice_index: int = 0,
    provider_id: str = "openai",
) -> FingerprintReferenceSnapshot:
    """A trusted-reference capture that always answers one fixed choice."""
    return build_fingerprint_snapshot(
        snapshot_id=snapshot_id,
        model_id=model_id,
        provider_id=provider_id,
        source_role=FingerprintSourceRole.TRUSTED_REFERENCE,
        suite=suite,
        repetitions=repetitions,
        observations=make_observations(suite, repetitions, choice_index=choice_index),
        captured_at=CAPTURED_AT,
    )


def make_observations_by_round(
    suite: FingerprintSuite,
    *,
    choice_index_by_round: Sequence[int],
    ref_prefix: str = "rounds",
) -> tuple[FingerprintSampleObservation, ...]:
    """One deterministic observation per (probe, round), choice chosen per round.

    Letting the answer depend on the round is what makes a capture able to
    answer one way in the first temporal window and another way in the second
    — the "one identity drifts against itself" shape Task 30's baseline is
    built from.
    """
    observations: list[FingerprintSampleObservation] = []
    for round_index, choice_index in enumerate(choice_index_by_round):
        for sequence_index, probe in enumerate(suite.probes):
            observations.append(
                FingerprintSampleObservation(
                    probe_id=probe.probe_id,
                    sequence_index=sequence_index,
                    round_index=round_index,
                    outcome=probe.choices[choice_index],
                    valid=True,
                    response_body_sha256="a" * 64,
                    evidence_ref=f"{ref_prefix}:{probe.probe_id}:{round_index}",
                )
            )
    return tuple(observations)


def make_reference_snapshot_with_rounds(
    *,
    suite: FingerprintSuite,
    snapshot_id: str,
    model_id: str,
    choice_index_by_round: Sequence[int],
    provider_id: str = "openai",
) -> FingerprintReferenceSnapshot:
    """A trusted-reference capture whose answer may change between rounds."""
    return build_fingerprint_snapshot(
        snapshot_id=snapshot_id,
        model_id=model_id,
        provider_id=provider_id,
        source_role=FingerprintSourceRole.TRUSTED_REFERENCE,
        suite=suite,
        repetitions=len(choice_index_by_round),
        observations=make_observations_by_round(suite, choice_index_by_round=choice_index_by_round),
        captured_at=CAPTURED_AT,
    )


@dataclass(frozen=True)
class FingerprintFixture:
    """A published, on-disk reference set plus the repository that holds it."""

    repository: FingerprintRepository
    set_path: Path
    reference_set: FingerprintReferenceSet
    suite: FingerprintSuite
    repetitions: int


def publish_reference_set(
    *,
    data_root: Path,
    suite: FingerprintSuite,
    snapshots: Sequence[FingerprintReferenceSnapshot],
    set_id: str = "test-identity-set",
    set_version: str = "0.1.0",
) -> FingerprintFixture:
    """Persist member snapshots, build the set, and write it to a readable path."""
    repository = FingerprintRepository.load(data_root=data_root)
    snapshot_sha256s: dict[str, str] = {}
    for snapshot in snapshots:
        repository.snapshots.save(snapshot)
        snapshot_sha256s[snapshot.snapshot_id] = repository.snapshots.snapshot_sha256(snapshot.snapshot_id)

    reference_set = FingerprintReferenceSetBuilder().build(
        fingerprint_set_id=set_id,
        fingerprint_set_version=set_version,
        snapshots=snapshots,
        snapshot_sha256s=snapshot_sha256s,
    )
    repository.sets.save(reference_set)

    set_path = data_root.parent / "fingerprint-reference-set.json"
    set_path.write_text(reference_set.model_dump_json(indent=2), encoding="utf-8")
    return FingerprintFixture(
        repository=repository,
        set_path=set_path,
        reference_set=reference_set,
        suite=suite,
        repetitions=snapshots[0].repetitions,
    )


@pytest.fixture
def fingerprint_reference(tmp_path: Path, suite: FingerprintSuite) -> FingerprintFixture:
    """A two-identity reference set captured with the STANDARD profile's rounds.

    The first identity carries the same model label the audited endpoint
    claims, so a claimed-reference distance exists; the second is a distinct
    impostor identity that ranks behind it.
    """
    repetitions = repetitions_for_profile(FingerprintProfile.STANDARD)
    snapshots = (
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
    )
    return publish_reference_set(data_root=tmp_path / "appdata", suite=suite, snapshots=snapshots)


def publish_validated_policy(
    fixture: FingerprintFixture,
    *,
    distance_threshold: float = 0.2,
    policy_id: str = "test-fingerprint-policy",
    policy_version: str = "0.1.0",
) -> FingerprintDecisionPolicy:
    """Publish a *validated* policy compatible with *fixture*'s reference set.

    The policy is a fixture standing in for a Task 18 held-out validation
    product; the gate under test is the runtime Rule 2 check, not the
    validation statistics themselves.
    """
    policy = build_fingerprint_policy(
        policy_id=policy_id,
        policy_version=policy_version,
        fingerprint_set_id=fixture.reference_set.fingerprint_set_id,
        fingerprint_set_content_sha256=fixture.reference_set.content_sha256,
        suite_content_sha256=fixture.suite.content_sha256,
        validated=True,
        minimum_comparable_probes=len(fixture.suite.probes),
        max_far_target=0.05,
        identity_count=2,
        held_out_capture_count=4,
        distance_threshold=distance_threshold,
        top1_accuracy=1.0,
        top3_accuracy=1.0,
        true_positive_rate=0.95,
        false_accept_rate=0.01,
    )
    fixture.repository.policies.save(policy)
    return policy


# ---------------------------------------------------------------------------
# HTTP mock + runner factory
# ---------------------------------------------------------------------------


def _completion_json(content: str, model: str = MODEL_ID) -> dict[str, object]:
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
    }


def _models_json(model: str = MODEL_ID) -> dict[str, object]:
    return {"object": "list", "data": [{"id": model, "object": "model"}]}


def mock_openai(
    mock: respx.MockRouter,
    *,
    suite: FingerprintSuite | None = None,
    choice_index: int = 0,
) -> None:
    """Register the two OpenAI routes the runner needs.

    With *suite* given, a request whose prompt is a fingerprint probe is
    answered with ``probe.choices[choice_index]`` (so the capture yields valid
    outcomes that either agree with or contradict a reference capture); every
    other request — protocol probes and Quick Suite items — gets an answer both
    benchmark graders can extract.
    """
    prompt_answers: dict[str, str] = {}
    if suite is not None:
        for probe in suite.probes:
            prompt_answers[probe.prompt] = probe.choices[choice_index]

    def responder(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        messages = payload.get("messages") or []
        answer = GRADABLE_ANSWER
        for message in messages:
            candidate = prompt_answers.get(message.get("content", ""))
            if candidate is not None:
                answer = candidate
                break
        return httpx.Response(200, json=_completion_json(answer), headers={"content-type": "application/json"})

    mock.get(f"{BASE_URL}/models").respond(status_code=200, json=_models_json())
    mock.post(f"{BASE_URL}/chat/completions").mock(side_effect=responder)


def mock_fingerprint_model(
    mock: respx.MockRouter,
    *,
    suite: FingerprintSuite,
    choice_index_by_round: Sequence[int] = (0,),
    invalid: bool = False,
    status_code: int = 200,
    response_model: str = MODEL_ID,
    response_model_by_request: Sequence[str] | None = None,
) -> None:
    """以 respx 注册一个 deterministic 的 OpenAI-compatible mock（Task 49）.

    五种必须能复现的行为（全部不依赖真实 OpenAI）：

    * **Model A / Model B** —— ``choice_index_by_round=(0,)`` / ``(1,)``：每个 probe
      永远答同一个 choice 索引；
    * **mixed / switching** —— 更长的 ``choice_index_by_round`` 把"第几轮"映射到
      choice 索引（末尾值对后续轮次重复）。某个 probe 的轮次由"该 prompt 已被请求
      多少次"推得，因此同一 probe 可以在第一个时间窗答 A、第二个时间窗答 B；
    * **invalid categorical output** —— ``invalid=True`` 对所有 probe 返回非 choice
      文本：非空但 normalizer 无法映射，观测被保留在保留的 invalid outcome 上；
    * **provider failure** —— ``status_code != 200`` 让每个补全请求失败，executor
      把样本记为 invalid 而不是缩小分母。

    ``response_model_by_request`` 让第 n 个补全响应的 ``model`` 字段循环取值 —— 即
    Scenario D 里 routing v2 要找的"显式 model split"。给了 ``suite`` 时只有 probe
    prompt 得到上述指纹答案；协议探测与 Quick Suite 条目仍拿到两个 grader 都能
    抽取的答案，因此整条 run 依然可评分。
    """
    rounds = tuple(choice_index_by_round) or (0,)
    probes_by_prompt = {probe.prompt: probe for probe in suite.probes}
    seen_per_probe: dict[str, int] = {}
    request_count = itertools.count()

    def responder(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        messages = payload.get("messages") or []
        index = next(request_count)
        if response_model_by_request:
            reported_model = response_model_by_request[index % len(response_model_by_request)]
        else:
            reported_model = response_model
        if status_code != 200:
            return httpx.Response(
                status_code,
                json={"error": {"message": "mock provider failure"}},
                headers={"content-type": "application/json"},
            )

        answer = GRADABLE_ANSWER
        for message in messages:
            probe = probes_by_prompt.get(message.get("content", ""))
            if probe is None:
                continue
            if invalid:
                answer = INVALID_MOCK_ANSWER
                break
            round_index = seen_per_probe.get(probe.probe_id, 0)
            seen_per_probe[probe.probe_id] = round_index + 1
            answer = probe.choices[rounds[min(round_index, len(rounds) - 1)]]
            break
        return httpx.Response(
            200,
            json=_completion_json(answer, model=reported_model),
            headers={"content-type": "application/json"},
        )

    advertised = response_model_by_request[0] if response_model_by_request else response_model
    mock.get(f"{BASE_URL}/models").respond(status_code=200, json=_models_json(advertised))
    mock.post(f"{BASE_URL}/chat/completions").mock(side_effect=responder)


def make_runner(
    config: AuditConfig,
    repository: RunArtifactRepository,
    **kwargs: object,
) -> UnifiedAuditRunner:
    """A UnifiedAuditRunner with the deterministic sandbox backend attached."""
    return UnifiedAuditRunner(
        config,
        api_key=API_KEY,
        target_id=TARGET_ID,
        repository=repository,
        code_backend=TrustedFakeBackend(),
        **kwargs,
    )
