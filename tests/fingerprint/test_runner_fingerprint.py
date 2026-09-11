"""Runner-level tests for the optional identity-evidence stage (Tasks 21–27).

Everything asserted here lives at the runner seam:

- the preflight gate that must fail *before* the first target request;
- the plan's fingerprint provenance and the request budget it reserves;
- the capture → snapshot → match → verification chain executed inside the one
  provider context, with evidence closure over the run's recorded evidence;
- cancellation propagation (a cancelled fingerprint stage never becomes a
  COMPLETED run);
- artifact backward-compatibility for runs that never asked for identity
  evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import respx

from llmtrace.config import AuditConfig
from llmtrace.execution.artifacts import RunArtifactRepository
from llmtrace.execution.models import UnifiedRunStatus
from llmtrace.execution.progress import (
    EVENT_CANCELLED,
    EVENT_PROGRESS,
    EVENT_STAGE,
    STAGE_FINGERPRINTING,
    CancellationToken,
    ProgressEvent,
)
from llmtrace.execution.runner import PreflightError, UnifiedAuditRunner
from llmtrace.fingerprint.executor import FINGERPRINT_EVIDENCE_TYPE
from llmtrace.fingerprint.models import (
    FingerprintMatchStatus,
    FingerprintProfile,
    FingerprintSourceRole,
    FingerprintSuite,
    repetitions_for_profile,
)

from .conftest import (
    BASE_URL,
    MODEL_ID,
    TARGET_ID,
    FingerprintFixture,
    make_reference_snapshot,
    make_runner,
    mock_openai,
    publish_reference_set,
    publish_validated_policy,
)

#: Protocol (4) + Quick Suite (32) requests, i.e. a run without identity evidence.
LEGACY_REQUESTS = 36
#: The built-in suite's probe count.
PROBE_COUNT = 6
STANDARD_REPETITIONS = repetitions_for_profile(FingerprintProfile.STANDARD)
FINGERPRINT_REQUESTS = PROBE_COUNT * STANDARD_REPETITIONS
#: Artifacts a run writes when identity evidence is off (Task 26).
LEGACY_ARTIFACTS = {
    "manifest.json",
    "report.json",
    "report.html",
    "capability_profile.json",
    "behavior_snapshot.json",
    "benchmark_runs.json",
}


def _fingerprint_runner(
    config: AuditConfig,
    repository: RunArtifactRepository,
    fixture: FingerprintFixture,
    **kwargs: object,
) -> UnifiedAuditRunner:
    """A runner with identity evidence switched on for *fixture*'s reference set."""
    return make_runner(
        config,
        repository,
        verify_model=True,
        fingerprint_profile=FingerprintProfile.STANDARD,
        fingerprint_set_path=fixture.set_path,
        fingerprint_repository=fixture.repository,
        **kwargs,
    )


class TestPreflightGate:
    """A broken fingerprint input is a hard failure with zero target requests."""

    @pytest.mark.asyncio
    async def test_verify_model_without_a_reference_set_sends_nothing(
        self, config: AuditConfig, api_key_env: None, tmp_path: Path
    ) -> None:
        repository = RunArtifactRepository(tmp_path)
        runner = make_runner(config, repository, verify_model=True)

        with respx.mock as mock:
            mock_openai(mock)
            with pytest.raises(PreflightError, match="--verify-model requires a fingerprint reference set"):
                await runner.run()
            # Preflight runs before any provider exists, so nothing left the process.
            assert mock.calls == []
        assert runner.request_budget is None

    @pytest.mark.asyncio
    async def test_unreadable_reference_set_fails_before_any_request(
        self,
        config: AuditConfig,
        api_key_env: None,
        tmp_path: Path,
        fingerprint_reference: FingerprintFixture,
    ) -> None:
        repository = RunArtifactRepository(tmp_path)
        runner = make_runner(
            config,
            repository,
            verify_model=True,
            fingerprint_set_path=tmp_path / "missing-fingerprint-set.json",
            fingerprint_repository=fingerprint_reference.repository,
        )

        with respx.mock as mock:
            mock_openai(mock)
            with pytest.raises(PreflightError, match="fingerprint reference set rejected"):
                await runner.run()
            assert mock.calls == []

    @pytest.mark.asyncio
    async def test_reference_set_captured_with_other_rounds_is_rejected(
        self,
        config: AuditConfig,
        api_key_env: None,
        tmp_path: Path,
        suite: FingerprintSuite,
    ) -> None:
        # A reference captured with the QUICK cost tier's rounds cannot be
        # compared against a STANDARD run: the distributions are not comparable.
        quick_repetitions = repetitions_for_profile(FingerprintProfile.QUICK)
        snapshots = (
            make_reference_snapshot(
                suite=suite,
                snapshot_id="ref-quick-capture-1",
                model_id=MODEL_ID,
                repetitions=quick_repetitions,
            ),
        )
        fixture = publish_reference_set(data_root=tmp_path / "appdata-quick", suite=suite, snapshots=snapshots)
        repository = RunArtifactRepository(tmp_path)
        runner = _fingerprint_runner(config, repository, fixture)

        with respx.mock as mock:
            mock_openai(mock)
            with pytest.raises(PreflightError, match="distributions would not be comparable"):
                await runner.run()
            assert mock.calls == []


class TestFingerprintRun:
    """The capture → snapshot → match → verification chain under one provider."""

    @pytest.mark.asyncio
    async def test_run_without_a_policy_ranks_but_never_produces_a_verdict(
        self,
        config: AuditConfig,
        api_key_env: None,
        tmp_path: Path,
        fingerprint_reference: FingerprintFixture,
        suite: FingerprintSuite,
    ) -> None:
        repository = RunArtifactRepository(tmp_path)
        events: list[ProgressEvent] = []
        runner = _fingerprint_runner(config, repository, fingerprint_reference, progress_sink=events.append)

        with respx.mock as mock:
            mock_openai(mock, suite=suite)
            result = await runner.run()

        # Ranking is available; a claim verdict is not (Rule 2).
        assert result.fingerprint_match is not None
        assert result.fingerprint_match.status == FingerprintMatchStatus.RANKED_ONLY
        assert result.fingerprint_verification is not None
        assert result.fingerprint_verification.claim_verdict_produced is False
        assert result.fingerprint_verification.policy is None
        # The most similar reference is evidence, never an identity claim (Rule 3).
        top = result.fingerprint_match.entries[0]
        assert top.model_id == MODEL_ID
        assert top.distance == pytest.approx(0.0)
        # The absence of a verdict is visible in the run's own warnings.
        assert any("no claim verdict" in warning for warning in result.warnings)
        assert result.status == UnifiedRunStatus.COMPLETED_WITH_WARNINGS

        # The capture is labelled as the audited endpoint, never as a reference.
        snapshot = result.fingerprint_snapshot
        assert snapshot is not None
        assert snapshot.source_role == FingerprintSourceRole.CANDIDATE_CAPTURE
        assert snapshot.provider_id == f"target:{TARGET_ID}"
        assert snapshot.model_id == MODEL_ID
        assert snapshot.snapshot_id == f"candidate-{result.execution_id}"
        assert snapshot.repetitions == STANDARD_REPETITIONS
        assert len(snapshot.observations) == FINGERPRINT_REQUESTS
        snapshot.verify_content_hash()

        # Evidence closure end to end: every observation points at evidence
        # this run actually recorded (Task 27).
        evidence_ids = {str(evidence.evidence_id) for evidence in result.evidence}
        assert {observation.evidence_ref for observation in snapshot.observations} <= evidence_ids
        fingerprint_evidence = [
            evidence for evidence in result.evidence if evidence.evidence_type == FINGERPRINT_EVIDENCE_TYPE
        ]
        assert len(fingerprint_evidence) == FINGERPRINT_REQUESTS

        # Fingerprint progress is reported through the runner's own emitter.
        stages = [event for event in events if event.type == EVENT_STAGE and event.stage == STAGE_FINGERPRINTING]
        assert len(stages) == 1
        assert stages[0].completed == 0
        assert stages[0].total == FINGERPRINT_REQUESTS
        progress = [event for event in events if event.type == EVENT_PROGRESS and event.stage == STAGE_FINGERPRINTING]
        assert [event.completed for event in progress] == list(range(1, FINGERPRINT_REQUESTS + 1))

        # Budget and artifacts cover protocol + benchmark + fingerprint.
        manifest = repository.load_manifest(result.execution_id)
        assert manifest.planned_requests == LEGACY_REQUESTS + FINGERPRINT_REQUESTS
        assert manifest.actual_requests == LEGACY_REQUESTS + FINGERPRINT_REQUESTS
        assert manifest.fingerprint_profile == FingerprintProfile.STANDARD.value
        assert manifest.fingerprint_set_id == fingerprint_reference.reference_set.fingerprint_set_id
        assert manifest.fingerprint_set_version == fingerprint_reference.reference_set.fingerprint_set_version
        assert manifest.fingerprint_set_content_sha256 == fingerprint_reference.reference_set.content_sha256

        run_dir = tmp_path / "runs" / result.execution_id
        assert (run_dir / "fingerprint_snapshot.json").exists()
        assert (run_dir / "fingerprint_match.json").exists()
        assert (run_dir / "fingerprint_verification.json").exists()

    @pytest.mark.asyncio
    async def test_validated_policy_produces_a_consistent_claim_verdict(
        self,
        config: AuditConfig,
        api_key_env: None,
        tmp_path: Path,
        fingerprint_reference: FingerprintFixture,
        suite: FingerprintSuite,
    ) -> None:
        publish_validated_policy(fingerprint_reference)
        repository = RunArtifactRepository(tmp_path)
        runner = _fingerprint_runner(config, repository, fingerprint_reference)

        with respx.mock as mock:
            mock_openai(mock, suite=suite)
            result = await runner.run()

        assert result.fingerprint_match is not None
        assert result.fingerprint_match.status == FingerprintMatchStatus.CONSISTENT_WITH_CLAIM
        assert result.fingerprint_match.claimed_reference_distance == pytest.approx(0.0)
        assert result.fingerprint_verification is not None
        assert result.fingerprint_verification.claim_verdict_produced is True
        assert result.fingerprint_verification.policy is not None
        assert result.fingerprint_verification.policy.validated is True
        # A validated policy leaves no "why not" note behind.
        assert result.status == UnifiedRunStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_contradicting_capture_is_inconsistent_while_ranking_another_model_first(
        self,
        config: AuditConfig,
        api_key_env: None,
        tmp_path: Path,
        fingerprint_reference: FingerprintFixture,
        suite: FingerprintSuite,
    ) -> None:
        publish_validated_policy(fingerprint_reference)
        repository = RunArtifactRepository(tmp_path)
        runner = _fingerprint_runner(config, repository, fingerprint_reference)

        with respx.mock as mock:
            # The endpoint answers the impostor's choice: the nearest reference
            # is 'other-model', yet the verdict is about the *claimed* model.
            mock_openai(mock, suite=suite, choice_index=1)
            result = await runner.run()

        assert result.fingerprint_match is not None
        assert result.fingerprint_match.entries[0].model_id == "other-model"
        assert result.fingerprint_match.status == FingerprintMatchStatus.INCONSISTENT_WITH_CLAIM
        assert result.fingerprint_match.claimed_model_id == MODEL_ID
        assert result.fingerprint_verification is not None
        assert result.fingerprint_verification.claim_verdict_produced is True


class TestFingerprintReportSection:
    """Task 44: the run's reports carry the machine-readable identity evidence."""

    @pytest.mark.asyncio
    async def test_reports_carry_the_fingerprint_section(
        self,
        config: AuditConfig,
        api_key_env: None,
        tmp_path: Path,
        fingerprint_reference: FingerprintFixture,
        suite: FingerprintSuite,
    ) -> None:
        publish_validated_policy(fingerprint_reference)
        repository = RunArtifactRepository(tmp_path)
        runner = _fingerprint_runner(config, repository, fingerprint_reference)

        with respx.mock as mock:
            mock_openai(mock, suite=suite)
            result = await runner.run()

        run_dir = tmp_path / "runs" / result.execution_id
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        assert report["schema_version"] == "1.4"

        section = report["fingerprint"]
        assert section["available"] is True
        assert section["experimental"] is True
        assert section["suite"]["suite_id"] == suite.suite_id
        assert section["suite"]["content_sha256"] == suite.content_sha256
        assert section["reference_set"]["fingerprint_set_id"] == fingerprint_reference.reference_set.fingerprint_set_id
        assert section["reference_set"]["content_sha256"] == fingerprint_reference.reference_set.content_sha256
        assert section["decision_policy"]["validated"] is True
        assert section["repetitions"] == STANDARD_REPETITIONS
        assert section["probe_coverage"]["compared_probes"] == PROBE_COUNT
        assert section["invalid_outcome_count"] == 0
        assert section["jsd_reference"]["basis"] == "claimed_reference"
        assert section["aggregate_jsd"] == pytest.approx(0.0)
        assert section["claimed_model_id"] == MODEL_ID
        assert section["verdict"]["match_status"] == FingerprintMatchStatus.CONSISTENT_WITH_CLAIM.value
        assert section["verdict"]["claim_verdict_produced"] is True

        # Routing v2 is wired but honest: no temporal fingerprint was captured,
        # so it is reported as unavailable rather than quietly assumed (Rule 2).
        assert section["routing"]["temporal_status"] == "unavailable"
        assert section["routing"]["item_count"] > 0
        assert any("temporal" in limitation for limitation in section["routing"]["limitations"])

        # Four independent confidence components, never one merged number.
        assert set(section["confidence"]) == {"measurement", "calibration", "fingerprint", "routing"}

        # The same data is rendered by the *existing* report template: one HTML
        # artifact, no separate fingerprint page (Task 44).
        html = (run_dir / "report.html").read_text(encoding="utf-8")
        assert "Model Fingerprint Evidence (Experimental)" in html
        assert "Behavioral Nearest References (Top-K)" in html
        assert "Routing Signals" in html
        assert "Confidence Bundle" in html
        assert "not cryptographic proof" in html
        assert [path.name for path in run_dir.iterdir() if path.suffix == ".html"] == ["report.html"]

    @pytest.mark.asyncio
    async def test_report_omits_the_section_when_the_stage_never_ran(
        self,
        config: AuditConfig,
        api_key_env: None,
        tmp_path: Path,
        fingerprint_reference: FingerprintFixture,
    ) -> None:
        # A protocol blocking failure skips the fingerprint stage entirely, so
        # there is no identity evidence to serialize and the reports keep the
        # pre-v0.6 shape (Rule 1).
        repository = RunArtifactRepository(tmp_path)
        runner = _fingerprint_runner(config, repository, fingerprint_reference)

        with respx.mock as mock:
            mock.post(f"{BASE_URL}/chat/completions").respond(status_code=401, json={"error": "Unauthorized"})
            result = await runner.run()

        assert result.fingerprint_snapshot is None
        assert any("fingerprint skipped" in warning for warning in result.warnings)

        run_dir = tmp_path / "runs" / result.execution_id
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        assert "fingerprint" not in report
        assert "Model Fingerprint Evidence" not in (run_dir / "report.html").read_text(encoding="utf-8")


class TestFingerprintCancellation:
    """Cancellation must reach the run as CANCELLED, never as COMPLETED (Task 25)."""

    @pytest.mark.asyncio
    async def test_cancel_at_the_fingerprint_stage_sends_no_probe(
        self,
        config: AuditConfig,
        api_key_env: None,
        tmp_path: Path,
        fingerprint_reference: FingerprintFixture,
        suite: FingerprintSuite,
    ) -> None:
        repository = RunArtifactRepository(tmp_path)
        events: list[ProgressEvent] = []
        token = CancellationToken()

        def sink(event: ProgressEvent) -> None:
            events.append(event)
            if event.type == EVENT_STAGE and event.stage == STAGE_FINGERPRINTING:
                token.cancel()

        runner = _fingerprint_runner(
            config,
            repository,
            fingerprint_reference,
            progress_sink=sink,
            cancel_token=token,
        )

        with respx.mock as mock:
            mock_openai(mock, suite=suite)
            result = await runner.run()

        assert result.status == UnifiedRunStatus.CANCELLED
        # Only protocol + benchmark were ever sent.
        assert runner.request_budget is not None
        assert runner.request_budget.consumed_requests == LEGACY_REQUESTS
        assert result.fingerprint_snapshot is None
        assert not (tmp_path / "runs" / result.execution_id).exists()
        assert events[-1].type == EVENT_CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_mid_capture_stops_at_the_next_request_boundary(
        self,
        config: AuditConfig,
        api_key_env: None,
        tmp_path: Path,
        fingerprint_reference: FingerprintFixture,
        suite: FingerprintSuite,
    ) -> None:
        repository = RunArtifactRepository(tmp_path)
        events: list[ProgressEvent] = []
        token = CancellationToken()
        completed_before_cancel = 3

        def sink(event: ProgressEvent) -> None:
            events.append(event)
            if event.type != EVENT_PROGRESS or event.stage != STAGE_FINGERPRINTING:
                return
            if event.completed == completed_before_cancel:
                token.cancel()

        runner = _fingerprint_runner(
            config,
            repository,
            fingerprint_reference,
            progress_sink=sink,
            cancel_token=token,
        )

        with respx.mock as mock:
            mock_openai(mock, suite=suite)
            result = await runner.run()

        assert result.status == UnifiedRunStatus.CANCELLED
        # The remaining probes were never sent.
        assert runner.request_budget is not None
        assert runner.request_budget.consumed_requests == LEGACY_REQUESTS + completed_before_cancel
        fingerprint_progress = [
            event for event in events if event.type == EVENT_PROGRESS and event.stage == STAGE_FINGERPRINTING
        ]
        assert len(fingerprint_progress) == completed_before_cancel
        assert result.fingerprint_snapshot is None
        assert not (tmp_path / "runs" / result.execution_id).exists()
        assert events[-1].type == EVENT_CANCELLED


class TestArtifactCompatibility:
    """A run that never asked for identity evidence is byte-for-byte the old run."""

    @pytest.mark.asyncio
    async def test_run_without_identity_evidence_writes_the_legacy_artifact_set(
        self, config: AuditConfig, api_key_env: None, tmp_path: Path
    ) -> None:
        repository = RunArtifactRepository(tmp_path)
        with respx.mock as mock:
            mock_openai(mock)
            result = await make_runner(config, repository).run()

        assert result.status == UnifiedRunStatus.COMPLETED
        assert result.fingerprint_snapshot is None
        assert result.fingerprint_match is None
        assert result.fingerprint_verification is None
        assert result.plan.fingerprint_requests == 0
        assert result.plan.fingerprint_profile is None

        run_dir = tmp_path / "runs" / result.execution_id
        assert {path.name for path in run_dir.iterdir()} == LEGACY_ARTIFACTS

        manifest = repository.load_manifest(result.execution_id)
        assert manifest.planned_requests == LEGACY_REQUESTS
        assert manifest.fingerprint_profile is None
        assert manifest.fingerprint_set_id is None
        assert manifest.fingerprint_set_version is None
        assert manifest.fingerprint_set_content_sha256 is None

    @pytest.mark.asyncio
    async def test_reference_set_is_ignored_while_identity_evidence_is_off(
        self,
        config: AuditConfig,
        api_key_env: None,
        tmp_path: Path,
        fingerprint_reference: FingerprintFixture,
    ) -> None:
        repository = RunArtifactRepository(tmp_path)
        runner = make_runner(
            config,
            repository,
            fingerprint_set_path=fingerprint_reference.set_path,
            fingerprint_repository=fingerprint_reference.repository,
        )

        with respx.mock as mock:
            mock_openai(mock)
            result = await runner.run()

        assert result.plan.fingerprint_requests == 0
        assert result.fingerprint_snapshot is None
        run_dir = tmp_path / "runs" / result.execution_id
        assert {path.name for path in run_dir.iterdir()} == LEGACY_ARTIFACTS
