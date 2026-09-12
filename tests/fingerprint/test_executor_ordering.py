"""Task 53 —— executor 的请求顺序：round interleaving 与 seeded shuffle.

``probe_order`` 存在的理由是避免"probe A 连续 N 次 → probe B 连续 N 次"把时间漂移
与 probe 身份混在一起。这里同时验证纯函数（顺序确定性、每轮都是全集的排列）与
executor 的观测顺序（每轮顺序、``sequence_index`` 每轮重置、同 seed 同序、不同 seed
可能变序）。

请求全部由 respx 拦截在 ``test.example.com``，不访问真实 endpoint。
"""

from __future__ import annotations

import pytest
import respx

from llmtrace.config import AuditConfig
from llmtrace.execution.budget import RequestBudget
from llmtrace.execution.evidence import InMemoryEvidenceRecorder
from llmtrace.fingerprint.executor import FingerprintExecutor
from llmtrace.fingerprint.models import FingerprintSampleObservation, FingerprintSuite
from llmtrace.fingerprint.suite import probe_order
from llmtrace.providers.factory import create_provider

from .conftest import API_KEY, mock_fingerprint_model

ROUNDS = 4
#: Seeds the ordering assertions are pinned to; the shuffle is a pure function of
#: ``seed + round_index``, so these outcomes are fully deterministic.
SEEDS = (0, 1, 2, 7)


def _probe_ids(suite: FingerprintSuite, *, seed: int, round_index: int) -> tuple[str, ...]:
    return tuple(probe.probe_id for probe in probe_order(suite.probes, seed=seed, round_index=round_index))


async def _observe(
    config: AuditConfig,
    suite: FingerprintSuite,
    *,
    seed: int,
    repetitions: int = ROUNDS,
) -> tuple[FingerprintSampleObservation, ...]:
    recorder = InMemoryEvidenceRecorder()
    budget = RequestBudget(len(suite.probes) * repetitions)
    provider = create_provider(config, API_KEY, evidence_recorder=recorder, request_budget=budget)
    with respx.mock as mock:
        mock_fingerprint_model(mock, suite=suite)
        async with provider:
            return await FingerprintExecutor(provider=provider, suite=suite).run(
                model=config.model,
                repetitions=repetitions,
                seed=seed,
            )


class TestProbeOrder:
    def test_the_same_seed_and_round_always_produce_the_same_order(self, suite: FingerprintSuite) -> None:
        first = probe_order(suite.probes, seed=3, round_index=2)
        second = probe_order(suite.probes, seed=3, round_index=2)

        assert [probe.probe_id for probe in first] == [probe.probe_id for probe in second]

    def test_every_round_is_a_permutation_of_the_probes(self, suite: FingerprintSuite) -> None:
        expected = {probe.probe_id for probe in suite.probes}
        for round_index in range(ROUNDS):
            ordered = probe_order(suite.probes, seed=0, round_index=round_index)
            assert len(ordered) == len(suite.probes)
            assert {probe.probe_id for probe in ordered} == expected

    def test_consecutive_rounds_use_a_different_order(self, suite: FingerprintSuite) -> None:
        # Round 0 and round 1 are shuffled by different RNG streams; equal orders
        # would mean the temporal windows share one probe sequence.
        assert _probe_ids(suite, seed=0, round_index=0) != _probe_ids(suite, seed=0, round_index=1)

    def test_a_different_seed_can_change_the_order(self, suite: FingerprintSuite) -> None:
        orders = {_probe_ids(suite, seed=seed, round_index=0) for seed in SEEDS}

        assert len(orders) > 1

    def test_the_input_sequence_is_not_mutated(self, suite: FingerprintSuite) -> None:
        original = [probe.probe_id for probe in suite.probes]

        probe_order(suite.probes, seed=0, round_index=3)

        assert [probe.probe_id for probe in suite.probes] == original


class TestExecutorOrdering:
    @pytest.mark.asyncio
    async def test_each_round_sends_the_probes_in_the_seeded_order(
        self,
        config: AuditConfig,
        api_key_env: None,
        suite: FingerprintSuite,
    ) -> None:
        observations = await _observe(config, suite, seed=0)

        for round_index in range(ROUNDS):
            observed = tuple(
                observation.probe_id for observation in observations if observation.round_index == round_index
            )
            assert observed == _probe_ids(suite, seed=0, round_index=round_index)

    @pytest.mark.asyncio
    async def test_sequence_index_restarts_at_every_round(
        self,
        config: AuditConfig,
        api_key_env: None,
        suite: FingerprintSuite,
    ) -> None:
        observations = await _observe(config, suite, seed=0)
        expected_positions = list(range(len(suite.probes)))

        for round_index in range(ROUNDS):
            positions = [
                observation.sequence_index for observation in observations if observation.round_index == round_index
            ]
            assert positions == expected_positions

    @pytest.mark.asyncio
    async def test_rounds_are_contiguous_and_positions_are_reshuffled(
        self,
        config: AuditConfig,
        api_key_env: None,
        suite: FingerprintSuite,
    ) -> None:
        observations = await _observe(config, suite, seed=0)

        # Round by round: never "probe A × rounds, then probe B × rounds".
        assert [observation.round_index for observation in observations] == sorted(
            observation.round_index for observation in observations
        )
        orders = {
            tuple(observation.probe_id for observation in observations if observation.round_index == round_index)
            for round_index in range(ROUNDS)
        }
        assert len(orders) > 1

    @pytest.mark.asyncio
    async def test_the_same_seed_reproduces_the_observation_order(
        self,
        config: AuditConfig,
        api_key_env: None,
        suite: FingerprintSuite,
    ) -> None:
        first = await _observe(config, suite, seed=5)
        second = await _observe(config, suite, seed=5)

        assert [(item.probe_id, item.round_index, item.sequence_index) for item in first] == [
            (item.probe_id, item.round_index, item.sequence_index) for item in second
        ]

    @pytest.mark.asyncio
    async def test_a_different_seed_changes_the_observation_order(
        self,
        config: AuditConfig,
        api_key_env: None,
        suite: FingerprintSuite,
    ) -> None:
        first = await _observe(config, suite, seed=0)
        second = await _observe(config, suite, seed=7)

        assert [item.probe_id for item in first] != [item.probe_id for item in second]

    @pytest.mark.asyncio
    async def test_every_request_is_still_sent_exactly_once_per_round(
        self,
        config: AuditConfig,
        api_key_env: None,
        suite: FingerprintSuite,
    ) -> None:
        observations = await _observe(config, suite, seed=0)

        assert len(observations) == len(suite.probes) * ROUNDS
        for round_index in range(ROUNDS):
            round_probes = [
                observation.probe_id for observation in observations if observation.round_index == round_index
            ]
            assert sorted(round_probes) == sorted(probe.probe_id for probe in suite.probes)
