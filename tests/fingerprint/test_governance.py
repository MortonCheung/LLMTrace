"""Task 46–48 governance tests for identity evidence.

Task 45's Web-compatibility assertion lives next to the service it protects
(``tests/service/test_run_service.py::test_result_view_tolerates_the_v06_report_schema``);
this module covers the other three gates of the v0.6 evidence foundation:

* **Task 46** — no API key ever reaches a fingerprint snapshot, reference set,
  decision policy, run artifact, SQLite file, JSON, or HTML.
* **Task 47** — probes are packaged synthetic public data: the only prompts
  that leave the process are the suite's, and no local file, chat history, or
  credential content is ever sent.
* **Task 48** — nothing in the source or in a generated report uses convicting
  language; the wording stays behavioral.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from llmtrace.config import AuditConfig
from llmtrace.execution.artifacts import RunArtifactRepository
from llmtrace.execution.budget import RequestBudget
from llmtrace.execution.evidence import InMemoryEvidenceRecorder
from llmtrace.execution.runner import UnifiedAuditRunner
from llmtrace.fingerprint.executor import FingerprintExecutor
from llmtrace.fingerprint.models import FingerprintMatchStatus, FingerprintProfile, FingerprintSuite
from llmtrace.fingerprint.suite import default_fingerprint_suite_path, load_fingerprint_suite
from llmtrace.providers.factory import create_provider

from .conftest import (
    API_KEY,
    BASE_URL,
    MODEL_ID,
    FingerprintFixture,
    make_runner,
    mock_openai,
    publish_validated_policy,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Task 48 —— 禁止的定罪式措辞（小写比较）.
BANNED_PHRASES = (
    "fake model",
    "fraud proven",
    "provider is cheating",
    "definitely model",
    "this is model",
)


def _completion_json(content: str) -> dict[str, object]:
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
    }


def _fingerprint_runner(
    config: AuditConfig,
    repository: RunArtifactRepository,
    fixture: FingerprintFixture,
) -> UnifiedAuditRunner:
    """A runner with identity evidence switched on for *fixture*'s reference set."""
    return make_runner(
        config,
        repository,
        verify_model=True,
        fingerprint_profile=FingerprintProfile.STANDARD,
        fingerprint_set_path=fixture.set_path,
        fingerprint_repository=fixture.repository,
    )


def _identity_evidence_sources() -> list[Path]:
    """The v0.6 identity-evidence surface whose wording Task 48 governs."""
    files: list[Path] = sorted((REPO_ROOT / "src" / "llmtrace" / "fingerprint").rglob("*.py"))
    files += sorted((REPO_ROOT / "src" / "llmtrace" / "fingerprint" / "resources").rglob("*.json"))
    files.append(REPO_ROOT / "src" / "llmtrace" / "reporting" / "fingerprint_report.py")
    files.append(REPO_ROOT / "src" / "llmtrace" / "reporting" / "templates" / "report.html.j2")
    files.append(REPO_ROOT / "src" / "llmtrace" / "reporting" / "console.py")
    files.append(REPO_ROOT / "src" / "llmtrace" / "analysis" / "confidence_v2.py")
    return [path for path in files if path.is_file()]


class TestSecretHygiene:
    """Task 46 — the API key stays in process memory."""

    @pytest.mark.asyncio
    async def test_api_key_never_reaches_any_artifact(
        self,
        config: AuditConfig,
        api_key_env: None,
        tmp_path: Path,
        fingerprint_reference: FingerprintFixture,
    ) -> None:
        publish_validated_policy(fingerprint_reference)
        repository = RunArtifactRepository(tmp_path)
        runner = _fingerprint_runner(config, repository, fingerprint_reference)

        with respx.mock as mock:
            mock_openai(mock)
            result = await runner.run()

        # The capture really happened, so the artifacts under test exist.
        assert result.fingerprint_snapshot is not None
        assert result.fingerprint_match is not None

        # Byte-level sweep: run artifacts, fingerprint snapshots / sets /
        # policies, the exported reference set, and any SQLite file.
        needle = API_KEY.encode("utf-8")
        leaked = sorted(
            str(path.relative_to(tmp_path))
            for path in tmp_path.rglob("*")
            if path.is_file() and needle in path.read_bytes()
        )
        assert leaked == []

        # ...and the serialized evidence itself carries no key.
        assert API_KEY not in json.dumps(result.fingerprint_snapshot.model_dump(mode="json"))
        assert API_KEY not in json.dumps(result.fingerprint_verification.model_dump(mode="json"))
        assert API_KEY not in " ".join(result.warnings)

        run_dir = tmp_path / "runs" / result.execution_id
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        assert API_KEY not in json.dumps(report["fingerprint"])
        assert API_KEY not in (run_dir / "report.html").read_text(encoding="utf-8")


class TestProbeProvenance:
    """Task 47 — probes are packaged synthetic public data."""

    def test_probes_are_the_packaged_public_suite(self, suite: FingerprintSuite) -> None:
        raw = json.loads(default_fingerprint_suite_path().read_text(encoding="utf-8"))
        assert [probe.prompt for probe in suite.probes] == [probe["prompt"] for probe in raw["probes"]]
        # Loading twice yields identical probe text: nothing is generated from
        # the local environment or from the audited endpoint.
        assert load_fingerprint_suite() == suite

    @pytest.mark.asyncio
    async def test_only_synthetic_probe_prompts_leave_the_process(
        self, config: AuditConfig, api_key_env: None, suite: FingerprintSuite
    ) -> None:
        repetitions = 2
        bodies: list[dict[str, object]] = []
        recorder = InMemoryEvidenceRecorder()
        budget = RequestBudget(len(suite.probes) * repetitions)

        def responder(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(
                200,
                json=_completion_json(suite.probes[0].choices[0]),
                headers={"content-type": "application/json"},
            )

        provider = create_provider(config, API_KEY, evidence_recorder=recorder, request_budget=budget)
        with respx.mock as mock:
            mock.post(f"{BASE_URL}/chat/completions").mock(side_effect=responder)
            async with provider:
                await FingerprintExecutor(provider=provider, suite=suite).run(
                    model=config.model, repetitions=repetitions
                )

        assert len(bodies) == len(suite.probes) * repetitions
        # Every request is exactly one user turn carrying a suite probe prompt.
        expected = {probe.prompt for probe in suite.probes}
        for body in bodies:
            messages = body["messages"]
            assert isinstance(messages, list) and len(messages) == 1
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] in expected

        # No local file, chat history, or credential content is ever in flight.
        serialized = json.dumps(bodies)
        assert API_KEY not in serialized
        assert "TEST_KEY" not in serialized
        assert str(REPO_ROOT) not in serialized


class TestThreatSemantics:
    """Task 48 — behavioral wording only, never a conviction."""

    def test_identity_evidence_sources_avoid_convicting_language(self) -> None:
        offenders: list[str] = []
        for path in _identity_evidence_sources():
            lowered = path.read_text(encoding="utf-8").lower()
            offenders += [f"{path.relative_to(REPO_ROOT)}: {phrase}" for phrase in BANNED_PHRASES if phrase in lowered]
        assert offenders == []

    def test_match_status_values_stay_behavioral(self) -> None:
        for status in FingerprintMatchStatus:
            assert not any(phrase in status.value.lower() for phrase in BANNED_PHRASES)
        assert FingerprintMatchStatus.CONSISTENT_WITH_CLAIM.value == "behavior_consistent_with_claim"
        assert FingerprintMatchStatus.INCONSISTENT_WITH_CLAIM.value == "behavior_inconsistent_with_claim"

    @pytest.mark.asyncio
    async def test_generated_reports_describe_behavior_not_guilt(
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
            # The mock answers each probe with the claimed reference's choice,
            # so the run reaches the "behavior consistent with claim" verdict.
            mock_openai(mock, suite=suite)
            result = await runner.run()

        run_dir = tmp_path / "runs" / result.execution_id
        json_text = (run_dir / "report.json").read_text(encoding="utf-8")
        html_text = (run_dir / "report.html").read_text(encoding="utf-8")

        for text in (json_text, html_text):
            lowered = text.lower()
            assert not any(phrase in lowered for phrase in BANNED_PHRASES)
            # 允许的措辞：行为学表述 + 非密码学证明声明。
            assert "not cryptographic proof" in lowered

        assert "behavior_consistent_with_claim" in json_text
