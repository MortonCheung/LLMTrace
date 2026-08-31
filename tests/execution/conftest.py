"""Shared fixtures for the execution-layer test suite."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from llmtrace.adapters.code_execution import CodeExecutionBackend, CodeExecutionResult
from llmtrace.config import AuditConfig, AuthStyle, Protocol
from llmtrace.execution.models import RunArtifactManifest, UnifiedRunStatus
from llmtrace.models.evidence import HTTPEvidence


@pytest.fixture
def config() -> AuditConfig:
    """Minimal OpenAI-protocol audit config."""
    return AuditConfig(
        protocol=Protocol.OPENAI,
        base_url="http://test.example.com/v1",
        model="my-real-model",
        api_key_env="TEST_KEY",
        auth_style=AuthStyle.AUTO,
        repeat_count=3,
        timeout=10.0,
        max_output_tokens=64,
        check_streaming=False,
        output_dir="reports",
    )


class TrustedFakeBackend(CodeExecutionBackend):
    """Deterministic, always-passing sandbox for tests (never executes code)."""

    def is_available(self) -> bool:
        return True

    def execute(self, code: str, *, timeout_seconds: float = 10.0) -> CodeExecutionResult:
        return CodeExecutionResult(success=True, stdout="OK", exit_code=0)


@pytest.fixture
def trusted_backend() -> CodeExecutionBackend:
    return TrustedFakeBackend()


def make_evidence(model: str = "my-real-model", status: int = 200) -> HTTPEvidence:
    """A minimal HTTP evidence object with a fresh UUID."""
    return HTTPEvidence(
        evidence_id=uuid.uuid4(),
        evidence_type="smoke_test",
        request_method="POST",
        request_url_redacted="https://redacted/",
        request_path="/v1/chat/completions",
        request_headers_redacted={},
        request_body_redacted={},
        request_model=model,
        response_model=model,
        response_text="The answer is 42.",
        http_status=status,
        total_latency_ms=50.0,
    )


def make_manifest(
    execution_id: str = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    *,
    target_id: str = "openai-test",
    candidate_model_id: str = "my-real-model",
    status: UnifiedRunStatus = UnifiedRunStatus.COMPLETED,
    created_at: datetime | None = None,
) -> RunArtifactManifest:
    return RunArtifactManifest(
        execution_id=execution_id,
        report_id="llmtrace_test",
        target_id=target_id,
        candidate_model_id=candidate_model_id,
        base_url_redacted="http://test.example.com/v1",
        protocol="openai",
        created_at=created_at or datetime.now(UTC),
        completed_at=datetime.now(UTC),
        status=status,
        suite_id="llmtrace_quick_v1",
        suite_version="0.1.0",
        adapter_id="llmtrace-quick-v1",
        adapter_version="0.1.0",
        scoring_policy_id="llmtrace_capability_v1",
        scoring_policy_version="0.1.0",
        generation_config_sha256="a" * 64,
        planned_requests=38,
        actual_requests=38,
    )
