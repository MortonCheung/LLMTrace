"""Task 27 — the fingerprint evidence closure.

Two halves of the same guarantee:

* ``assert_evidence_closure`` must reject any observation pointing at evidence
  this run never recorded (no fabricated refs may reach an artifact);
* the executor must actually *create* that evidence through the provider — one
  real request, one recorded ``fingerprint_probe`` evidence, one observation —
  and must never shrink the denominator when a request fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import respx

from llmtrace.config import AuditConfig
from llmtrace.execution.budget import RequestBudget
from llmtrace.execution.evidence import InMemoryEvidenceRecorder
from llmtrace.fingerprint.executor import (
    FINGERPRINT_EVIDENCE_TYPE,
    FingerprintEvidenceClosureError,
    FingerprintExecutor,
    assert_evidence_closure,
)
from llmtrace.fingerprint.models import INVALID_OUTCOME, FingerprintSuite
from llmtrace.models.evidence import EvidenceType
from llmtrace.providers.factory import create_provider

from .conftest import API_KEY, BASE_URL, GRADABLE_ANSWER, make_observations, mock_openai


class TestClosureValidator:
    def test_accepts_refs_that_the_run_recorded(self, suite: FingerprintSuite) -> None:
        observations = make_observations(suite, 2)
        recorded = {observation.evidence_ref for observation in observations}

        # A superset (protocol / benchmark evidence) is still a closed run.
        assert_evidence_closure(observations, recorded | {"some-other-evidence-id"})

    def test_rejects_a_fabricated_ref(self, suite: FingerprintSuite) -> None:
        observations = make_observations(suite, 2)
        recorded = {observation.evidence_ref for observation in observations}
        recorded.discard(observations[0].evidence_ref)

        with pytest.raises(FingerprintEvidenceClosureError) as excinfo:
            assert_evidence_closure(observations, recorded)

        assert excinfo.value.error_code == "FINGERPRINT_EVIDENCE_CLOSURE_ERROR"
        assert observations[0].evidence_ref in str(excinfo.value)

    def test_rejects_every_ref_when_nothing_was_recorded(self, suite: FingerprintSuite) -> None:
        with pytest.raises(FingerprintEvidenceClosureError):
            assert_evidence_closure(make_observations(suite, 1), set())

    def test_empty_observations_close_vacuously(self) -> None:
        assert_evidence_closure((), set())


async def _run_executor(
    config: AuditConfig,
    suite: FingerprintSuite,
    *,
    repetitions: int,
    recorder: InMemoryEvidenceRecorder,
    budget: RequestBudget,
    progress_sink: object = None,
) -> tuple[object, ...]:
    provider = create_provider(config, API_KEY, evidence_recorder=recorder, request_budget=budget)
    async with provider:
        return await FingerprintExecutor(
            provider=provider,
            suite=suite,
            progress_sink=progress_sink,  # type: ignore[arg-type]
        ).run(model=config.model, repetitions=repetitions)


class TestExecutorEvidence:
    @pytest.mark.asyncio
    async def test_every_probe_request_becomes_recorded_evidence(
        self, config: AuditConfig, api_key_env: None, suite: FingerprintSuite
    ) -> None:
        repetitions = 2
        recorder = InMemoryEvidenceRecorder()
        budget = RequestBudget(len(suite.probes) * repetitions)

        with respx.mock as mock:
            mock_openai(mock, suite=suite)
            observations = await _run_executor(config, suite, repetitions=repetitions, recorder=recorder, budget=budget)

        assert len(observations) == len(suite.probes) * repetitions
        assert budget.consumed_requests == len(observations)
        assert all(evidence.evidence_type == FINGERPRINT_EVIDENCE_TYPE for evidence in recorder.list())
        assert {evidence.evidence_type for evidence in recorder.list()} == {EvidenceType.FINGERPRINT_PROBE.value}

        # The mock answers each probe with its first choice → all valid.
        assert all(observation.valid for observation in observations)
        assert {observation.outcome for observation in observations} == {probe.choices[0] for probe in suite.probes}

        # The real closure check the runner performs on a candidate capture.
        assert_evidence_closure(observations, {str(evidence.evidence_id) for evidence in recorder.list()})

    @pytest.mark.asyncio
    async def test_ungradable_output_is_invalid_but_never_dropped(
        self, config: AuditConfig, api_key_env: None, suite: FingerprintSuite
    ) -> None:
        repetitions = 2
        recorder = InMemoryEvidenceRecorder()
        budget = RequestBudget(len(suite.probes) * repetitions)

        with respx.mock as mock:
            # No probe answers registered → every response is a non-choice.
            mock_openai(mock)
            observations = await _run_executor(config, suite, repetitions=repetitions, recorder=recorder, budget=budget)

        assert GRADABLE_ANSWER not in {observation.outcome for observation in observations}
        assert len(observations) == len(suite.probes) * repetitions
        assert all(not observation.valid for observation in observations)
        assert {observation.outcome for observation in observations} == {INVALID_OUTCOME}
        # The denominator equals the requests actually sent.
        assert budget.consumed_requests == len(observations)

    @pytest.mark.asyncio
    async def test_failed_request_is_recorded_as_invalid_not_raised(
        self, config: AuditConfig, api_key_env: None, suite: FingerprintSuite
    ) -> None:
        repetitions = 2
        recorder = InMemoryEvidenceRecorder()
        budget = RequestBudget(len(suite.probes) * repetitions)

        with respx.mock as mock:
            mock.post(f"{BASE_URL}/chat/completions").respond(status_code=500, json={"error": "boom"})
            observations = await _run_executor(config, suite, repetitions=repetitions, recorder=recorder, budget=budget)

        assert len(observations) == len(suite.probes) * repetitions
        assert all(not observation.valid for observation in observations)
        assert len(recorder) == len(observations)
        assert_evidence_closure(observations, {str(evidence.evidence_id) for evidence in recorder.list()})

    @pytest.mark.asyncio
    async def test_repetitions_below_one_fails_before_any_request(
        self, config: AuditConfig, api_key_env: None, suite: FingerprintSuite
    ) -> None:
        recorder = InMemoryEvidenceRecorder()
        budget = RequestBudget(len(suite.probes))
        provider = create_provider(config, API_KEY, evidence_recorder=recorder, request_budget=budget)

        with respx.mock:
            async with provider:
                executor = FingerprintExecutor(provider=provider, suite=suite)
                with pytest.raises(ValueError, match="repetitions must be >= 1"):
                    await executor.run(model=config.model, repetitions=0)

        assert budget.consumed_requests == 0


class TestExecutorProgress:
    @pytest.mark.asyncio
    async def test_progress_reaches_the_full_probe_total(
        self, config: AuditConfig, api_key_env: None, suite: FingerprintSuite
    ) -> None:
        from llmtrace.execution.progress import EVENT_PROGRESS, STAGE_FINGERPRINTING

        repetitions = 2
        total = len(suite.probes) * repetitions
        events: list[object] = []
        recorder = InMemoryEvidenceRecorder()
        budget = RequestBudget(total)

        with respx.mock as mock:
            mock_openai(mock, suite=suite)
            await _run_executor(
                config,
                suite,
                repetitions=repetitions,
                recorder=recorder,
                budget=budget,
                progress_sink=events.append,
            )

        assert len(events) == total
        assert all(event.type == EVENT_PROGRESS for event in events)
        assert all(event.stage == STAGE_FINGERPRINTING for event in events)
        assert [event.completed for event in events] == list(range(1, total + 1))
        assert all(event.total == total for event in events)


def test_closure_error_module_is_reachable() -> None:
    # Guard against an accidental rename: the error code is part of the
    # fail-closed contract the runner's warning path relies on.
    assert FingerprintEvidenceClosureError.error_code == "FINGERPRINT_EVIDENCE_CLOSURE_ERROR"
    assert Path("tests/fingerprint").exists()
