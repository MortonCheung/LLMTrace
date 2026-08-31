"""Tests for the unified execution planner (execution/planner.py)."""

from __future__ import annotations

import pytest

from llmtrace.config import AuditConfig, Protocol
from llmtrace.execution.planner import (
    build_unified_execution_plan,
    derive_target_id,
    sanitize_target_id,
)


class TestTargetId:
    def test_derive_is_stable(self) -> None:
        assert derive_target_id("openai", "http://example.com/v1") == derive_target_id(
            "openai", "http://example.com/v1/"
        )

    def test_derive_differs_by_protocol(self) -> None:
        assert derive_target_id("openai", "http://example.com/v1") != derive_target_id(
            "anthropic", "http://example.com/v1"
        )

    def test_derive_does_not_leak_credential(self) -> None:
        # Userinfo credentials and secret query values must never surface in
        # the derived target id.
        result = derive_target_id("openai", "http://user:supersecretpw@example.com/v1?api_key=abc123")
        assert "supersecretpw" not in result
        assert "user" not in result
        assert "abc123" not in result

    def test_sanitize_strips(self) -> None:
        assert sanitize_target_id("  my-target  ") == "my-target"

    def test_sanitize_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            sanitize_target_id("   ")

    def test_sanitize_rejects_path_unsafe(self) -> None:
        for bad in ("a/b", "a\\b", "..", ".", "a\0b"):
            with pytest.raises(ValueError):
                sanitize_target_id(bad)


class TestUnifiedExecutionPlan:
    def _config(self, **overrides: object) -> AuditConfig:
        kwargs: dict[str, object] = {
            "protocol": Protocol.OPENAI,
            "base_url": "http://test.example.com/v1",
            "model": "demo-model",
            "api_key_env": "TEST_KEY",
            "repeat_count": 3,
            "max_output_tokens": 64,
            "check_streaming": False,
        }
        kwargs.update(overrides)
        return AuditConfig(**kwargs)  # type: ignore[arg-type]

    def _plan(self, **overrides: object):
        return build_unified_execution_plan(self._config(**overrides), target_id="openai-test")

    def test_benchmark_always_32(self) -> None:
        assert self._plan().benchmark_requests == 32

    def test_streaming_off_protocol_count(self) -> None:
        # connectivity(1) + catalog(1) + baseline(3) + invalid(1) = 6
        plan = self._plan(check_streaming=False)
        assert plan.protocol_probe_requests == 6
        assert plan.planned_requests == 38

    def test_streaming_on_protocol_count(self) -> None:
        # + streaming(2) = 8
        plan = self._plan(check_streaming=True)
        assert plan.protocol_probe_requests == 8
        assert plan.planned_requests == 40

    def test_repeat_max_protocol_count(self) -> None:
        plan = self._plan(repeat_count=10, check_streaming=False)
        # connectivity(1) + catalog(1) + baseline(10) + invalid(1) = 13
        assert plan.protocol_probe_requests == 13

    def test_total_exact_and_maximum(self) -> None:
        plan = self._plan(check_streaming=False)
        assert plan.planned_requests == plan.protocol_probe_requests + plan.benchmark_requests
        assert plan.maximum_requests == plan.planned_requests

    def test_cost_is_none(self) -> None:
        assert self._plan().estimated_cost is None

    def test_output_token_ceiling(self) -> None:
        # 32 × 512 (benchmark) + 5 × 64 (protocol completions; the catalog GET
        # consumes no output tokens) = 16384 + 320
        plan = self._plan(check_streaming=False)
        assert plan.maximum_output_token_ceiling == 32 * 512 + 5 * 64

    def test_plan_is_deterministic(self) -> None:
        a = self._plan()
        b = self._plan()
        assert a.plan_id == b.plan_id
        assert a.generation_config_sha256 == b.generation_config_sha256

    def test_generation_config_hash_matches_canonical(self) -> None:
        import hashlib
        import json

        from llmtrace.adapters.quick_suite import get_quick_suite_generation_config

        config = get_quick_suite_generation_config()
        expected = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        assert self._plan().generation_config_sha256 == expected

    def test_requires_secure_sandbox(self) -> None:
        assert self._plan().requires_secure_code_sandbox is True
