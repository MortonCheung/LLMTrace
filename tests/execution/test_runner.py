"""Integration tests for the UnifiedAuditRunner (execution/runner.py)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
import respx

from llmtrace.adapters.code_execution import CodeExecutionBackend, CodeExecutionResult
from llmtrace.config import AuditConfig, Protocol
from llmtrace.execution import runner as runner_module
from llmtrace.execution.artifacts import RunArtifactRepository, sha256_of
from llmtrace.execution.models import UnifiedRunStatus
from llmtrace.execution.runner import PreflightError, UnifiedAuditRunner
from llmtrace.providers.factory import create_provider
from llmtrace.providers.openai_compatible import OpenAICompatibleProvider

from .conftest import TrustedFakeBackend

API_KEY = "sk-super-secret-123"
TARGET_ID = "openai-test-target"


def _completion_json(content: str, model: str = "my-real-model") -> dict[str, object]:
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


def _models_json(model: str = "my-real-model") -> dict[str, object]:
    return {"object": "list", "data": [{"id": model, "object": "model"}]}


@pytest.fixture
def api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", API_KEY)


@pytest.fixture
def config() -> AuditConfig:
    return AuditConfig(
        protocol=Protocol.OPENAI,
        base_url="http://test.example.com/v1",
        model="my-real-model",
        api_key_env="TEST_KEY",
        repeat_count=1,
        max_output_tokens=64,
        check_streaming=False,
        output_dir="reports",
    )


def _mock_openai(respx_mock: respx.MockRouter) -> None:
    # The answer must be gradable by *all four* Quick Suite graders:
    # "(A)" → ARC letter extraction, "42" → GSM8K numeric extraction,
    # HumanEval/IFEval are graded by the fake backend / constraint checker.
    respx_mock.get("http://test.example.com/v1/models").respond(status_code=200, json=_models_json())
    respx_mock.post("http://test.example.com/v1/chat/completions").respond(
        status_code=200,
        json=_completion_json("The answer is (A). The answer is 42."),
        headers={"content-type": "application/json"},
    )


# ---------------------------------------------------------------------------
# Provider-level evidence recording + budget
# ---------------------------------------------------------------------------


class TestProviderEvidenceRecording:
    @pytest.mark.asyncio
    async def test_every_request_recorded_exactly_once(self, config: AuditConfig, api_key_env: None) -> None:
        from llmtrace.execution.evidence import InMemoryEvidenceRecorder

        recorder = InMemoryEvidenceRecorder()
        with respx.mock as mock:
            _mock_openai(mock)
            provider = create_provider(config, API_KEY, evidence_recorder=recorder)
            async with provider:
                await provider.complete(config.model, [{"role": "user", "content": "hi"}])
                await provider.list_models()
                await provider.complete(config.model, [{"role": "user", "content": "hi"}])

        # 3 real requests → 3 distinct evidence records, in arrival order.
        assert len(recorder) == 3
        ids = [str(e.evidence_id) for e in recorder.list()]
        assert len(set(ids)) == 3

    @pytest.mark.asyncio
    async def test_http_error_recorded_once(self, config: AuditConfig, api_key_env: None) -> None:
        from llmtrace.execution.evidence import InMemoryEvidenceRecorder

        recorder = InMemoryEvidenceRecorder()
        with respx.mock as mock:
            mock.post("http://test.example.com/v1/chat/completions").respond(status_code=500, json={"error": "boom"})
            provider = create_provider(config, API_KEY, evidence_recorder=recorder)
            async with provider:
                evidence = await provider.complete(config.model, [{"role": "user", "content": "hi"}])

        assert evidence.http_status == 500
        assert len(recorder) == 1

    @pytest.mark.asyncio
    async def test_budget_enforced_before_send(self, config: AuditConfig, api_key_env: None) -> None:
        from llmtrace.execution.budget import RequestBudget, RequestBudgetExceededError

        budget = RequestBudget(1)
        with respx.mock as mock:
            _mock_openai(mock)
            provider = create_provider(config, API_KEY, request_budget=budget)
            async with provider:
                await provider.complete(config.model, [{"role": "user", "content": "hi"}])
                with pytest.raises(RequestBudgetExceededError):
                    await provider.complete(config.model, [{"role": "user", "content": "hi"}])

        # Only one request actually left the process.
        assert budget.consumed_requests == 1


# ---------------------------------------------------------------------------
# Unified runner — happy path and red lines
# ---------------------------------------------------------------------------


class TestUnifiedAuditRunner:
    def _runner(self, config: AuditConfig, repo: RunArtifactRepository, **kwargs: object) -> UnifiedAuditRunner:
        return UnifiedAuditRunner(
            config,
            api_key=API_KEY,
            target_id=TARGET_ID,
            repository=repo,
            code_backend=TrustedFakeBackend(),
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_happy_path_full_pipeline(self, config: AuditConfig, api_key_env: None, tmp_path: Path) -> None:
        repo = RunArtifactRepository(tmp_path)
        with respx.mock as mock:
            _mock_openai(mock)
            result = await self._runner(config, repo).run()

        assert result.status == UnifiedRunStatus.COMPLETED
        assert result.protocol_audit is not None
        assert result.capability_profile is not None
        assert result.behavior_snapshot is not None
        # A fully gradable mock answer must yield a healthy measurement:
        # 32/32 graded, no failure, no ungradable.
        assert result.measurement_summary is not None
        assert result.measurement_summary.total_item_count == 32
        assert result.measurement_summary.graded_item_count == 32
        assert result.measurement_summary.failure_item_count == 0
        assert result.measurement_summary.ungradable_item_count == 0
        # First run → no baseline.
        assert result.behavior_drift is None
        assert result.reference_comparison is None

        # Artifacts were committed.
        run_dir = tmp_path / "runs" / result.execution_id
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "report.json").exists()
        assert (run_dir / "report.html").exists()
        assert (run_dir / "capability_profile.json").exists()
        assert (run_dir / "behavior_snapshot.json").exists()
        assert (run_dir / "benchmark_runs.json").exists()

        # protocol (connectivity 1 + catalog 1 + baseline 1 + invalid 1) + 32 = 36.
        manifest = repo.load_manifest(result.execution_id)
        assert manifest.planned_requests == 36
        assert manifest.actual_requests == 36

    @pytest.mark.asyncio
    async def test_no_test_model_anywhere(self, config: AuditConfig, api_key_env: None, tmp_path: Path) -> None:
        repo = RunArtifactRepository(tmp_path)
        with respx.mock as mock:
            _mock_openai(mock)
            result = await self._runner(config, repo).run()

        request_models = {e.request_model for e in result.evidence}
        assert "test-model" not in request_models
        # The Quick Suite's 32 items must all use the declared model.
        assert result.behavior_snapshot is not None
        assert len(result.behavior_snapshot.items) == 32

        # Every Quick Suite evidence (traced via item evidence_refs) uses the
        # declared model, never a hardcoded test-model.
        evidence_by_id = {str(e.evidence_id): e for e in result.evidence}
        for item in result.behavior_snapshot.items:
            for ref in item.evidence_refs:
                assert evidence_by_id[ref].request_model == "my-real-model"

    @pytest.mark.asyncio
    async def test_api_key_never_serialized(self, config: AuditConfig, api_key_env: None, tmp_path: Path) -> None:
        repo = RunArtifactRepository(tmp_path)
        with respx.mock as mock:
            _mock_openai(mock)
            result = await self._runner(config, repo).run()

        run_dir = tmp_path / "runs" / result.execution_id
        for artifact in run_dir.iterdir():
            if artifact.suffix == ".html" or artifact.suffix == ".json":
                assert API_KEY not in artifact.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_protocol_blocking_skips_benchmark(
        self, config: AuditConfig, api_key_env: None, tmp_path: Path
    ) -> None:
        repo = RunArtifactRepository(tmp_path)
        with respx.mock as mock:
            mock.post("http://test.example.com/v1/chat/completions").respond(
                status_code=401, json={"error": "Unauthorized"}
            )
            result = await self._runner(config, repo).run()

        assert result.status == UnifiedRunStatus.PARTIAL
        assert result.capability_profile is None
        assert result.behavior_snapshot is None
        assert result.benchmark_runs == ()
        # Only the connectivity probe (1 request) was sent; no benchmark waste.
        manifest = repo.load_manifest(result.execution_id)
        assert manifest.actual_requests <= 1

    @pytest.mark.asyncio
    async def test_wall_timeout_never_completed(self, config: AuditConfig, api_key_env: None, tmp_path: Path) -> None:
        repo = RunArtifactRepository(tmp_path)

        async def _slow(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(5)
            return httpx.Response(200, json=_completion_json("42"))

        with respx.mock as mock:
            mock.post("http://test.example.com/v1/chat/completions").mock(side_effect=_slow)
            mock.get("http://test.example.com/v1/models").respond(status_code=200, json=_models_json())
            result = await self._runner(config, repo, max_wall_seconds=0.05).run()

        assert result.status == UnifiedRunStatus.FAILED
        assert "wall-clock" in " ".join(result.warnings)

    @pytest.mark.asyncio
    async def test_history_auto_baseline_no_self_compare(
        self, config: AuditConfig, api_key_env: None, tmp_path: Path
    ) -> None:
        repo = RunArtifactRepository(tmp_path)
        with respx.mock as mock:
            _mock_openai(mock)
            first = await self._runner(config, repo).run()

        assert first.behavior_drift is None

        # Second run against the same target/model automatically finds the
        # first run's snapshot and compares — but never against itself.
        with respx.mock as mock:
            _mock_openai(mock)
            second = await self._runner(config, repo).run()

        assert second.behavior_drift is not None
        manifest = repo.load_manifest(second.execution_id)
        assert manifest.baseline_execution_id == first.execution_id
        assert manifest.baseline_execution_id != second.execution_id
        # Identical behavior → no significant drift.
        assert second.behavior_drift.drift_level.value == "NO_SIGNIFICANT_DRIFT"

    @pytest.mark.asyncio
    async def test_different_model_not_compared(self, config: AuditConfig, api_key_env: None, tmp_path: Path) -> None:
        repo = RunArtifactRepository(tmp_path)
        with respx.mock as mock:
            _mock_openai(mock)
            await self._runner(config, repo).run()

        # A run against a *different* declared model must never be treated as
        # the same object's cross-run drift. The repository filters it out.
        other_config = config.model_copy(update={"model": "other-model"})
        with respx.mock as mock:
            _mock_openai(mock)
            second = await self._runner(other_config, repo).run()

        assert second.behavior_drift is None
        # Not an error — just no comparable baseline; the run completes cleanly.
        assert second.status == UnifiedRunStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_incompatible_history_skipped_with_warning(
        self, config: AuditConfig, api_key_env: None, tmp_path: Path
    ) -> None:
        repo = RunArtifactRepository(tmp_path)
        with respx.mock as mock:
            _mock_openai(mock)
            first = await self._runner(config, repo).run()

        # Tamper the prior run's generation config hash so the compatibility
        # gate must reject it — the engine, not the repository, is the final
        # authority. The manifest hash is updated to match so the tampering
        # passes *integrity* verification and reaches the engine's
        # compatibility gate (SHA mismatch is covered by its own test below).
        run_dir = tmp_path / "runs" / first.execution_id
        snap_path = run_dir / "behavior_snapshot.json"
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
        snap["generation_config_sha256"] = "b" * 64
        tampered = json.dumps(snap)
        snap_path.write_text(tampered, encoding="utf-8")
        manifest_path = run_dir / "manifest.json"
        manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_json["artifacts"]["behavior_snapshot.json"] = sha256_of(tampered)
        manifest_path.write_text(json.dumps(manifest_json), encoding="utf-8")

        with respx.mock as mock:
            _mock_openai(mock)
            second = await self._runner(config, repo).run()

        assert second.behavior_drift is None
        assert any("skipped incompatible baseline" in w for w in second.warnings)

    @pytest.mark.asyncio
    async def test_explicit_baseline_snapshot(self, config: AuditConfig, api_key_env: None, tmp_path: Path) -> None:
        repo = RunArtifactRepository(tmp_path)
        with respx.mock as mock:
            _mock_openai(mock)
            first = await self._runner(config, repo).run()

        baseline_path = tmp_path / "runs" / first.execution_id / "behavior_snapshot.json"

        with respx.mock as mock:
            _mock_openai(mock)
            second = await self._runner(config, repo, baseline_snapshot_path=baseline_path, compare_latest=False).run()

        # The explicit baseline is compared via the same engine, producing a
        # real drift result against identical behavior.
        assert second.behavior_drift is not None
        assert second.behavior_drift.drift_level.value == "NO_SIGNIFICANT_DRIFT"

    @pytest.mark.asyncio
    async def test_incompatible_reference_no_fake_comparison(
        self, config: AuditConfig, api_key_env: None, tmp_path: Path
    ) -> None:
        from datetime import UTC, datetime

        from llmtrace.scoring.models import (
            CapabilityDimension,
            CapabilityProfile,
            DimensionScoreResult,
            DimensionScoreStatus,
        )
        from llmtrace.scoring.reference import ReferenceProvenance, ReferenceSnapshot

        # A reference with a mismatched suite id must be rejected by the
        # comparator gate — no fake delta, but the current run is preserved.
        dims = tuple(
            DimensionScoreResult(
                dimension=d,
                status=DimensionScoreStatus.UNCALIBRATED,
                raw_normalized_score=1.0,
            )
            for d in CapabilityDimension
        )
        reference = ReferenceSnapshot(
            snapshot_id="incompatible-ref",
            model_id="gpt-x",
            provider_id="openai",
            created_at=datetime(2026, 8, 10, tzinfo=UTC),
            suite_id="llmtrace_quick_v999",
            suite_version="0.1.0",
            capability_profile=CapabilityProfile(
                scoring_policy_id="llmtrace-capability-v1",
                scoring_policy_version="0.1.0",
                dimensions=dims,
                coverage_weight=0.75,
            ),
            provenance=ReferenceProvenance(
                source_type="benchmark_run",
                created_by="llmtrace",
                created_at=datetime(2026, 8, 10, tzinfo=UTC),
                suite_sha256="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
                benchmark_revision="quick-v1-rev",
                runner_version="0.3.0",
            ),
        )
        ref_path = tmp_path / "reference.json"
        ref_path.write_text(reference.model_dump_json(), encoding="utf-8")

        repo = RunArtifactRepository(tmp_path)
        with respx.mock as mock:
            _mock_openai(mock)
            result = await self._runner(config, repo, reference_snapshot_path=ref_path).run()

        # No fake comparison, run preserved, and a clear diagnostic warning.
        assert result.reference_comparison is None
        assert result.capability_profile is not None
        assert any("reference comparison skipped" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Quick Suite declared-model propagation (real provider, contract layer)
# ---------------------------------------------------------------------------


class TestQuickSuiteDeclaredModel:
    @pytest.mark.asyncio
    async def test_quick_suite_uses_provider_config_model(self, config: AuditConfig, api_key_env: None) -> None:
        from llmtrace.execution.quick_suite import QuickSuiteRunner

        with respx.mock as mock:
            _mock_openai(mock)
            provider = OpenAICompatibleProvider(config, API_KEY)
            runner = QuickSuiteRunner(provider, code_backend=TrustedFakeBackend())
            async with provider:
                result = await runner.run()

        # 4 tasks × 8 items = 32 items, all with declared-model evidence.
        all_items = [item for run in result.run_results for item in run.task_attempts[0].item_results]
        assert len(all_items) == 32
        # The provider's evidence_recorder path is exercised by the runner;
        # here we assert the canonical per-task provenance is preserved.
        assert len(result.run_results) == 4
        source_ids = {run.source_id for run in result.run_results}
        assert source_ids == {"arc_challenge", "humaneval", "gsm8k", "ifeval"}

    @pytest.mark.asyncio
    async def test_default_backend_never_in_process(self) -> None:
        from llmtrace.adapters.code_execution import (
            DockerCodeExecutionBackend,
            _InProcessExecutionBackend,
            create_code_execution_backend,
        )

        # The production factory must never silently return the unsafe
        # in-process backend. When Docker is missing it raises; when Docker is
        # present it returns the Docker backend.
        try:
            backend = create_code_execution_backend()
        except Exception as exc:  # SandboxUnavailableError on Docker-less CI
            assert "in-process" in str(exc).lower()
            return
        assert isinstance(backend, DockerCodeExecutionBackend)
        assert not isinstance(backend, _InProcessExecutionBackend)


# ---------------------------------------------------------------------------
# PR #16 merge-blocker regressions
# ---------------------------------------------------------------------------


class _LifecycleSpy:
    """Wraps a provider and counts context-manager entries/exits."""

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self.entered = 0
        self.exited = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    async def __aenter__(self) -> _LifecycleSpy:
        self.entered += 1
        await self._inner.__aenter__()  # type: ignore[attr-defined]
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.exited += 1
        await self._inner.__aexit__(*exc_info)  # type: ignore[attr-defined]


class _UnavailableBackend(CodeExecutionBackend):
    """A sandbox that reports itself unavailable (preflight must fail)."""

    def is_available(self) -> bool:
        return False

    def execute(self, code: str, *, timeout_seconds: float = 10.0) -> CodeExecutionResult:
        raise RuntimeError("sandbox unavailable")


class TestMergeBlockerRegressions:
    """Targeted regressions for the five merge blockers of PR #16."""

    def _runner(self, config: AuditConfig, repo: RunArtifactRepository, **kwargs: object) -> UnifiedAuditRunner:
        return UnifiedAuditRunner(
            config,
            api_key=API_KEY,
            target_id=TARGET_ID,
            repository=repo,
            code_backend=TrustedFakeBackend(),
            **kwargs,
        )

    # -- Blocker 1: Provider lifecycle ---------------------------------------

    @pytest.mark.asyncio
    async def test_provider_lifecycle_single_owner(
        self, config: AuditConfig, api_key_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = RunArtifactRepository(tmp_path)
        real_create = runner_module.create_provider
        spies: list[_LifecycleSpy] = []

        def spy_factory(*args: object, **kwargs: object) -> _LifecycleSpy:
            spy = _LifecycleSpy(real_create(*args, **kwargs))  # type: ignore[arg-type]
            spies.append(spy)
            return spy

        monkeypatch.setattr(runner_module, "create_provider", spy_factory)
        with respx.mock as mock:
            _mock_openai(mock)
            result = await self._runner(config, repo).run()

        # One provider instance for the whole run — never one per stage.
        assert len(spies) == 1
        # ...opened exactly once and closed exactly once: the runner is the
        # single lifecycle owner for protocol + benchmark.
        assert spies[0].entered == 1
        assert spies[0].exited == 1
        # The benchmark really executed through that same open provider.
        assert result.measurement_summary is not None
        assert result.measurement_summary.total_item_count == 32

    # -- Blocker 2: response-side secret echo ---------------------------------

    @pytest.mark.asyncio
    async def test_response_echo_never_persisted(self, config: AuditConfig, api_key_env: None, tmp_path: Path) -> None:
        repo = RunArtifactRepository(tmp_path)
        body = _completion_json("The answer is (A). The answer is 42.")
        body["echoed_secret"] = API_KEY  # adversarial echo in the response body
        with respx.mock as mock:
            mock.get("http://test.example.com/v1/models").respond(
                status_code=200, json=_models_json(), headers={"x-debug": API_KEY}
            )
            mock.post("http://test.example.com/v1/chat/completions").respond(
                status_code=200,
                json=body,
                headers={"x-debug": API_KEY, "content-type": "application/json"},
            )
            result = await self._runner(config, repo).run()

        assert result.status == UnifiedRunStatus.COMPLETED
        run_dir = tmp_path / "runs" / result.execution_id
        for artifact in run_dir.iterdir():
            assert API_KEY not in artifact.read_text(encoding="utf-8"), f"leak in {artifact.name}"
        assert all(API_KEY not in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_error_message_echo_never_persisted(
        self, config: AuditConfig, api_key_env: None, tmp_path: Path
    ) -> None:
        repo = RunArtifactRepository(tmp_path)
        with respx.mock as mock:
            mock.post("http://test.example.com/v1/chat/completions").respond(
                status_code=401, json={"error": f"Unauthorized key {API_KEY}"}
            )
            result = await self._runner(config, repo).run()

        assert result.status == UnifiedRunStatus.PARTIAL
        run_dir = tmp_path / "runs" / result.execution_id
        for artifact in run_dir.iterdir():
            assert API_KEY not in artifact.read_text(encoding="utf-8"), f"leak in {artifact.name}"
        assert all(API_KEY not in w for w in result.warnings)

    # -- Blocker 3: history artifact integrity --------------------------------

    @pytest.mark.asyncio
    async def test_tampered_history_sha_skips_with_warning(
        self, config: AuditConfig, api_key_env: None, tmp_path: Path
    ) -> None:
        repo = RunArtifactRepository(tmp_path)
        with respx.mock as mock:
            _mock_openai(mock)
            first = await self._runner(config, repo).run()

        # Tamper the prior snapshot WITHOUT updating the manifest: the
        # recorded SHA no longer matches the artifact on disk.
        snap_path = tmp_path / "runs" / first.execution_id / "behavior_snapshot.json"
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
        snap["generation_config_sha256"] = "b" * 64
        snap_path.write_text(json.dumps(snap), encoding="utf-8")

        with respx.mock as mock:
            _mock_openai(mock)
            second = await self._runner(config, repo).run()

        # The tampered baseline is skipped with an integrity warning — never
        # silently compared, never crashing the current run.
        assert second.behavior_drift is None
        assert any("skipped corrupted historical baseline" in w for w in second.warnings)
        assert second.status == UnifiedRunStatus.COMPLETED_WITH_WARNINGS

    # -- Blocker 4: preflight before any HTTP request -------------------------

    @pytest.mark.asyncio
    async def test_preflight_sandbox_unavailable_before_http(
        self, config: AuditConfig, api_key_env: None, tmp_path: Path
    ) -> None:
        repo = RunArtifactRepository(tmp_path)
        # No routes registered: any HTTP request would raise inside respx.
        with respx.mock, pytest.raises(PreflightError) as excinfo:
            await UnifiedAuditRunner(
                config,
                api_key=API_KEY,
                target_id=TARGET_ID,
                repository=repo,
                code_backend=_UnavailableBackend(),
            ).run()
        assert "sandbox" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_preflight_unwritable_artifact_root(
        self, config: AuditConfig, api_key_env: None, tmp_path: Path
    ) -> None:
        repo = RunArtifactRepository(tmp_path)
        tmp_path.chmod(0o500)
        try:
            with respx.mock, pytest.raises(PreflightError):
                await self._runner(config, repo).run()
        finally:
            tmp_path.chmod(0o700)

    @pytest.mark.asyncio
    async def test_preflight_missing_baseline_snapshot(
        self, config: AuditConfig, api_key_env: None, tmp_path: Path
    ) -> None:
        repo = RunArtifactRepository(tmp_path)
        with respx.mock, pytest.raises(PreflightError):
            await self._runner(config, repo, baseline_snapshot_path=tmp_path / "missing.json").run()

    # -- Blocker 5: status reflects measurement health ------------------------

    @pytest.mark.asyncio
    async def test_zero_graded_items_is_partial(self, config: AuditConfig, api_key_env: None, tmp_path: Path) -> None:
        repo = RunArtifactRepository(tmp_path)
        post_calls = {"n": 0}

        async def _responder(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200, json=_models_json())
            post_calls["n"] += 1
            if post_calls["n"] <= 3:
                # The 3 protocol POST probes (connectivity, baseline,
                # invalid-model) succeed...
                return httpx.Response(200, json=_completion_json("The answer is (A). The answer is 42."))
            # ...then every benchmark request fails: 0 of 32 items measured.
            return httpx.Response(500, json={"error": "boom"})

        with respx.mock as mock:
            mock.get("http://test.example.com/v1/models").mock(side_effect=_responder)
            mock.post("http://test.example.com/v1/chat/completions").mock(side_effect=_responder)
            result = await self._runner(config, repo).run()

        assert result.status == UnifiedRunStatus.PARTIAL
        assert result.measurement_summary is not None
        assert result.measurement_summary.graded_item_count == 0
        assert result.measurement_summary.failure_item_count == 32
        assert any("measurement unavailable" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_degraded_measurement_warns(self, config: AuditConfig, api_key_env: None, tmp_path: Path) -> None:
        repo = RunArtifactRepository(tmp_path)
        with respx.mock as mock:
            mock.get("http://test.example.com/v1/models").respond(status_code=200, json=_models_json())
            # An ungradable answer (no letter, no number) degrades the
            # measurement without losing it entirely.
            mock.post("http://test.example.com/v1/chat/completions").respond(
                status_code=200, json=_completion_json("I do not know.")
            )
            result = await self._runner(config, repo).run()

        assert result.status == UnifiedRunStatus.COMPLETED_WITH_WARNINGS
        assert result.measurement_summary is not None
        m = result.measurement_summary
        assert m.graded_item_count + m.ungradable_item_count == m.total_item_count
        assert m.graded_item_count > 0
        assert m.ungradable_item_count > 0
        assert any("measurement degraded" in w for w in result.warnings)

    # -- Blocker 6: plan identity must not depend on secrets ------------------

    def test_plan_id_secret_invariant(self, config: AuditConfig) -> None:
        from llmtrace.execution.planner import build_unified_execution_plan

        def _plan_id(base_url: str) -> str:
            cfg = config.model_copy(update={"base_url": base_url})
            return build_unified_execution_plan(cfg, target_id=TARGET_ID).plan_id

        # Different credentials / secret query values → same plan identity.
        assert _plan_id("http://user:secret-a@host.example.com/v1") == _plan_id(
            "http://user:secret-b@host.example.com/v1"
        )
        assert _plan_id("http://host.example.com/v1?api_key=aaa") == _plan_id("http://host.example.com/v1?api_key=bbb")
        # A genuinely different endpoint must still be a different identity.
        assert _plan_id("http://host.example.com/v1") != _plan_id("http://host.example.com/v2")
