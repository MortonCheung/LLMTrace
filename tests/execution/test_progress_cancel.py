"""Vertical tests for v0.5 progress events and cooperative cancellation.

Covers the minimal runner observability surface added for the web app:
- a ``progress_sink`` receives a deterministic stage / item timeline;
- a pre-cancelled ``CancellationToken`` yields a CANCELLED result with no
  report artifacts;
- cancelling mid-benchmark lands at the next item boundary, keeps the partial
  progress already reported, and never degrades cancellation into per-item
  FAILURE semantics.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import respx

from llmtrace.config import AuditConfig, Protocol
from llmtrace.execution.artifacts import RunArtifactRepository
from llmtrace.execution.models import UnifiedRunStatus
from llmtrace.execution.progress import (
    EVENT_CANCELLED,
    EVENT_DONE,
    EVENT_PROGRESS,
    EVENT_STAGE,
    STAGE_BENCHMARK,
    STAGE_CANCELLED,
    STAGE_DONE,
    STAGE_PROTOCOL,
    STAGE_REPORTING,
    CancellationToken,
    ProgressEvent,
)
from llmtrace.execution.runner import UnifiedAuditRunner

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


def _mock_openai(respx_mock: respx.MockRouter) -> None:
    respx_mock.get("http://test.example.com/v1/models").respond(status_code=200, json=_models_json())
    respx_mock.post("http://test.example.com/v1/chat/completions").respond(
        status_code=200,
        json=_completion_json("The answer is (A). The answer is 42."),
        headers={"content-type": "application/json"},
    )


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


@pytest.fixture
def api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", API_KEY)


def _runner(
    config: AuditConfig,
    repo: RunArtifactRepository,
    events: list[ProgressEvent],
    token: CancellationToken | None = None,
) -> UnifiedAuditRunner:
    return UnifiedAuditRunner(
        config,
        api_key=API_KEY,
        target_id=TARGET_ID,
        repository=repo,
        code_backend=TrustedFakeBackend(),
        progress_sink=events.append,
        cancel_token=token,
    )


class TestProgressSink:
    @pytest.mark.asyncio
    async def test_full_run_emits_complete_timeline(
        self, config: AuditConfig, api_key_env: None, tmp_path: Path
    ) -> None:
        repo = RunArtifactRepository(tmp_path)
        events: list[ProgressEvent] = []
        with respx.mock as mock:
            _mock_openai(mock)
            result = await _runner(config, repo, events).run()

        assert result.status == UnifiedRunStatus.COMPLETED
        # Terminal success event was emitted.
        assert events[-1].type == EVENT_DONE
        assert events[-1].stage == STAGE_DONE
        # Stage events cover protocol → benchmark → ... → done.
        stages = {e.stage for e in events if e.type == EVENT_STAGE}
        assert STAGE_PROTOCOL in stages
        assert STAGE_BENCHMARK in stages
        assert STAGE_REPORTING in stages
        # Benchmark progress reached the full 32/32.
        assert any(e.type == EVENT_PROGRESS and e.completed == 32 and e.total == 32 for e in events)
        # Request accounting rides along (protocol requests seen by benchmark).
        benchmark_starts = [e for e in events if e.type == EVENT_STAGE and e.stage == STAGE_BENCHMARK]
        assert benchmark_starts and benchmark_starts[0].requests >= 1

    @pytest.mark.asyncio
    async def test_sink_defaults_to_noop_for_cli(
        self, config: AuditConfig, api_key_env: None, tmp_path: Path
    ) -> None:
        repo = RunArtifactRepository(tmp_path)
        with respx.mock as mock:
            _mock_openai(mock)
            # No progress_sink / cancel_token passed → legacy CLI semantics.
            result = await UnifiedAuditRunner(
                config,
                api_key=API_KEY,
                target_id=TARGET_ID,
                repository=repo,
                code_backend=TrustedFakeBackend(),
            ).run()
        assert result.status == UnifiedRunStatus.COMPLETED


class TestCooperativeCancellation:
    @pytest.mark.asyncio
    async def test_precancelled_returns_cancelled_without_artifacts(
        self, config: AuditConfig, api_key_env: None, tmp_path: Path
    ) -> None:
        repo = RunArtifactRepository(tmp_path)
        events: list[ProgressEvent] = []
        token = CancellationToken()
        token.cancel()

        with respx.mock as mock:
            _mock_openai(mock)
            result = await _runner(config, repo, events, token).run()

        assert result.status == UnifiedRunStatus.CANCELLED
        # No protocol/benchmark requests ever fired.
        assert len(result.evidence) == 0
        # No report artifacts for a cancelled run.
        assert not (tmp_path / "runs" / result.execution_id).exists()
        assert events[-1].type == EVENT_CANCELLED
        assert events[-1].stage == STAGE_CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_at_item_boundary_keeps_partial_progress(
        self, config: AuditConfig, api_key_env: None, tmp_path: Path
    ) -> None:
        repo = RunArtifactRepository(tmp_path)
        events: list[ProgressEvent] = []
        token = CancellationToken()

        def sink(event: ProgressEvent) -> None:
            events.append(event)
            # Cancel as soon as the first benchmark item finishes; the next
            # item boundary must observe it and stop cleanly.
            if event.type == EVENT_PROGRESS and event.stage == STAGE_BENCHMARK:
                token.cancel()

        with respx.mock as mock:
            _mock_openai(mock)
            result = await UnifiedAuditRunner(
                config,
                api_key=API_KEY,
                target_id=TARGET_ID,
                repository=repo,
                code_backend=TrustedFakeBackend(),
                progress_sink=sink,
                cancel_token=token,
            ).run()

        assert result.status == UnifiedRunStatus.CANCELLED
        # Exactly the items finished before the cancel were reported.
        progress = [e for e in events if e.type == EVENT_PROGRESS]
        assert len(progress) == 1
        assert progress[0].completed == 1
        assert progress[0].total == 32
        # Cancellation is not a measurement failure: no FAILURE items were
        # manufactured and no report artifacts exist.
        assert not (tmp_path / "runs" / result.execution_id).exists()
        assert events[-1].type == EVENT_CANCELLED
