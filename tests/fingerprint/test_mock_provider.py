"""Task 49 —— deterministic Mock Provider 的五种行为.

每一种行为都通过**真实的** ``FingerprintExecutor`` → ``LLMProvider`` →
``RequestBudget`` → ``EvidenceRecorder`` 路径跑一遍（Rule 4），只把 socket 换成
respx；没有任何用例访问真实 OpenAI（``BASE_URL`` 是 ``test.example.com``，未注册的
路由在 respx 下会直接报错而不是出网）。

覆盖：Fingerprint Model A / Model B、mixed / switching behavior、invalid
categorical output、provider failure。
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import respx

from llmtrace.config import AuditConfig
from llmtrace.execution.budget import RequestBudget
from llmtrace.execution.evidence import InMemoryEvidenceRecorder
from llmtrace.fingerprint.executor import FingerprintExecutor
from llmtrace.fingerprint.models import (
    INVALID_OUTCOME,
    FingerprintSampleObservation,
    FingerprintSuite,
)
from llmtrace.providers.factory import create_provider

from .conftest import API_KEY, mock_fingerprint_model

REPETITIONS = 3


async def _capture(
    config: AuditConfig,
    suite: FingerprintSuite,
    *,
    repetitions: int = REPETITIONS,
    choice_index_by_round: Sequence[int] = (0,),
    invalid: bool = False,
    status_code: int = 200,
) -> tuple[tuple[FingerprintSampleObservation, ...], InMemoryEvidenceRecorder, RequestBudget]:
    """在 mock 之下跑一次真实采集，返回观测 / 证据记录器 / 预算."""
    expected = len(suite.probes) * repetitions
    recorder = InMemoryEvidenceRecorder()
    budget = RequestBudget(expected)
    provider = create_provider(config, API_KEY, evidence_recorder=recorder, request_budget=budget)
    with respx.mock as mock:
        mock_fingerprint_model(
            mock,
            suite=suite,
            choice_index_by_round=choice_index_by_round,
            invalid=invalid,
            status_code=status_code,
        )
        async with provider:
            observations = await FingerprintExecutor(provider=provider, suite=suite).run(
                model=config.model,
                repetitions=repetitions,
            )
    return observations, recorder, budget


def _by_round(
    observations: tuple[FingerprintSampleObservation, ...], round_index: int
) -> tuple[FingerprintSampleObservation, ...]:
    return tuple(observation for observation in observations if observation.round_index == round_index)


class TestDeterministicMockIdentities:
    """Fingerprint Model A / Model B：一个 identity 永远答同一个 choice."""

    @pytest.mark.parametrize("choice_index", [0, 1])
    @pytest.mark.asyncio
    async def test_a_fixed_choice_identity_answers_only_that_choice(
        self,
        config: AuditConfig,
        api_key_env: None,
        suite: FingerprintSuite,
        choice_index: int,
    ) -> None:
        observations, recorder, budget = await _capture(config, suite, choice_index_by_round=(choice_index,))

        expected = len(suite.probes) * REPETITIONS
        assert len(observations) == expected
        assert all(observation.valid for observation in observations)
        assert {observation.outcome for observation in observations} == {
            probe.choices[choice_index] for probe in suite.probes
        }
        # 每个 identity 的分布是 one-hot，且每个样本都对应一条真实证据。
        assert budget.consumed_requests == expected
        assert len(recorder) == expected

    @pytest.mark.asyncio
    async def test_two_identities_are_distinguishable_by_their_answers(
        self,
        config: AuditConfig,
        api_key_env: None,
        suite: FingerprintSuite,
    ) -> None:
        model_a, _, _ = await _capture(config, suite, choice_index_by_round=(0,))
        model_b, _, _ = await _capture(config, suite, choice_index_by_round=(1,))

        outcomes_a = {observation.outcome for observation in model_a}
        outcomes_b = {observation.outcome for observation in model_b}
        assert outcomes_a.isdisjoint(outcomes_b)


class TestSwitchingBehaviour:
    """mixed / switching：同一 probe 在不同轮次回答不同 choice."""

    @pytest.mark.asyncio
    async def test_the_answer_follows_the_round(
        self,
        config: AuditConfig,
        api_key_env: None,
        suite: FingerprintSuite,
    ) -> None:
        observations, _, _ = await _capture(config, suite, choice_index_by_round=(0, 1))

        first, second = _by_round(observations, 0), _by_round(observations, 1)
        assert len(first) == len(second) == len(suite.probes)
        assert {observation.outcome for observation in first} == {probe.choices[0] for probe in suite.probes}
        assert {observation.outcome for observation in second} == {probe.choices[1] for probe in suite.probes}
        # 混答仍然全部有效：分母没有因为切换而缩小。
        assert all(observation.valid for observation in observations)

    @pytest.mark.asyncio
    async def test_the_last_round_index_repeats_for_any_later_round(
        self,
        config: AuditConfig,
        api_key_env: None,
        suite: FingerprintSuite,
    ) -> None:
        observations, _, _ = await _capture(config, suite, choice_index_by_round=(0, 1))

        for round_index in (1, 2):
            assert {observation.outcome for observation in _by_round(observations, round_index)} == {
                probe.choices[1] for probe in suite.probes
            }


class TestInvalidCategoricalOutput:
    @pytest.mark.asyncio
    async def test_ungradable_categorical_output_is_retained(
        self,
        config: AuditConfig,
        api_key_env: None,
        suite: FingerprintSuite,
    ) -> None:
        observations, recorder, budget = await _capture(config, suite, invalid=True)

        expected = len(suite.probes) * REPETITIONS
        assert len(observations) == expected
        assert all(not observation.valid for observation in observations)
        assert {observation.outcome for observation in observations} == {INVALID_OUTCOME}
        # invalid 样本仍然各自对应一条真实证据，不是被丢弃的请求。
        assert budget.consumed_requests == expected
        assert len(recorder) == expected


class TestProviderFailure:
    @pytest.mark.asyncio
    async def test_a_failing_endpoint_keeps_every_sample(
        self,
        config: AuditConfig,
        api_key_env: None,
        suite: FingerprintSuite,
    ) -> None:
        observations, recorder, budget = await _capture(config, suite, status_code=500)

        expected = len(suite.probes) * REPETITIONS
        assert len(observations) == expected
        assert all(not observation.valid for observation in observations)
        assert {observation.outcome for observation in observations} == {INVALID_OUTCOME}
        assert budget.consumed_requests == expected
        assert len(recorder) == expected
        assert all(observation.evidence_ref for observation in observations)
