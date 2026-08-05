"""证据链与风险判断单元测试.

覆盖：
- 稳定性探针只引用 baseline 证据
- 模型标识漂移独立形成 HIGH 风险
- 原始字节哈希（非流式 + 流式）
- request_id 大小写不敏感提取与 response_id 分离
"""

from __future__ import annotations

import hashlib
import json

import httpx
import pytest
import respx

from llmtrace.analysis.risk import analyze_risk
from llmtrace.config import AuditConfig, Protocol
from llmtrace.models.audit import RiskLevel
from llmtrace.models.evidence import HTTPEvidence
from llmtrace.models.findings import ProbeStatus, Severity
from llmtrace.probes.baseline import BaselineProbe
from llmtrace.probes.stability import StabilityProbe
from llmtrace.providers.base import _extract_request_id
from llmtrace.providers.openai_compatible import OpenAICompatibleProvider

API_KEY = "test-api-key-12345"


def _make_config() -> AuditConfig:
    """构造测试配置."""
    return AuditConfig(
        protocol=Protocol.OPENAI,
        base_url="http://test.example.com",
        model="mock-model-v1",
        api_key_env="TEST_KEY",
        timeout=10.0,
        max_output_tokens=64,
    )


def _make_evidence(
    evidence_type: str,
    response_model: str | None = "mock-model-v1",
    http_status: int | None = 200,
    exception: tuple[str, str] | None = None,
) -> HTTPEvidence:
    """构造带指定类型的证据."""
    ev = HTTPEvidence(
        request_method="POST",
        request_url_redacted="http://test.example.com/v1/chat/completions",
        request_path="/v1/chat/completions",
        request_headers_redacted={"authorization": "Bearer ***"},
        evidence_type=evidence_type,
        http_status=http_status,
        response_model=response_model,
    )
    if exception:
        ev.exception_type, ev.exception_message = exception
    return ev


class TestStabilityBaselineIsolation:
    """9.1 稳定性探针只分析 baseline 证据."""

    def test_analyze_only_references_baseline_evidence(self) -> None:
        """混合多类型证据时，稳定性结果只引用 3 条 baseline 的 evidence_id."""
        baseline_evs = [
            _make_evidence("baseline", response_model="mock-model-v1"),
            _make_evidence("baseline", response_model="mock-model-v1"),
            _make_evidence("baseline", response_model="mock-model-v1"),
        ]
        other_evs = [
            _make_evidence("streaming_comparison", response_model="mock-model-v1"),
            _make_evidence("streaming_baseline", response_model="mock-model-v1"),
            _make_evidence("invalid_model", response_model=None, http_status=404),
        ]
        evidence_list = [*baseline_evs, *other_evs]

        probe = StabilityProbe(_make_config(), None)  # type: ignore[arg-type]
        finding = probe.analyze(evidence_list)

        expected_refs = {str(e.evidence_id) for e in baseline_evs}
        assert len(finding.evidence_refs) == 3
        assert set(finding.evidence_refs) == expected_refs
        # 任何引用都不指向非 baseline 证据
        assert not ({str(e.evidence_id) for e in other_evs} & set(finding.evidence_refs))

    def test_analyze_ignores_invalid_and_streaming_for_success_rate(self) -> None:
        """稳定性的成功率只基于 baseline，不受其他类型证据影响."""
        baseline_evs = [
            _make_evidence("baseline", response_model="mock-model-v1"),
            _make_evidence("baseline", response_model="mock-model-v1"),
            _make_evidence("baseline", response_model="mock-model-v1"),
        ]
        # 无效模型被正常拒绝、流式对照成功，均不应影响稳定性
        other_evs = [
            _make_evidence("invalid_model", response_model=None, http_status=404),
            _make_evidence("streaming_comparison", response_model="mock-model-v1"),
            _make_evidence("streaming_baseline", response_model="mock-model-v1"),
        ]

        probe = StabilityProbe(_make_config(), None)  # type: ignore[arg-type]
        finding = probe.analyze([*baseline_evs, *other_evs])

        assert finding.status == ProbeStatus.PASS
        assert any("基线请求数: 3" in fact for fact in finding.facts), finding.facts


class TestModelDriftRisk:
    """9.2 模型标识漂移独立形成 HIGH 风险."""

    def test_stability_analyze_drift_high(self) -> None:
        """baseline 返回 v1/v2/v1 时，稳定性探针产出 FAIL/HIGH，且不含无效模型证据."""
        baseline_evs = [
            _make_evidence("baseline", response_model="mock-model-v1"),
            _make_evidence("baseline", response_model="mock-model-v2"),
            _make_evidence("baseline", response_model="mock-model-v1"),
        ]

        probe = StabilityProbe(_make_config(), None)  # type: ignore[arg-type]
        finding = probe.analyze(baseline_evs)

        assert finding.status == ProbeStatus.FAIL
        assert finding.severity == Severity.HIGH
        assert any("返回模型不一致" in inf for inf in finding.inferences), finding.inferences
        assert analyze_risk([finding]) == RiskLevel.HIGH

    @pytest.mark.asyncio
    async def test_baseline_probe_drift_high(self) -> None:
        """BaselineProbe 在会话内收到多个不相关模型标识时直接产出 FAIL/HIGH."""
        responses = [
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": "mock-model-v1",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "a"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
            {
                "id": "chatcmpl-2",
                "object": "chat.completion",
                "model": "mock-model-v2",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "b"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
            {
                "id": "chatcmpl-3",
                "object": "chat.completion",
                "model": "mock-model-v1",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "c"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        ]

        with respx.mock as mock:
            route = mock.post("http://test.example.com/v1/chat/completions")
            route.side_effect = [httpx.Response(200, json=r) for r in responses]

            config = _make_config()
            config.repeat_count = 3
            async with OpenAICompatibleProvider(config, API_KEY) as provider:
                probe = BaselineProbe(config, provider)
                outcome = await probe.run()

        finding = outcome.findings[0]
        assert finding.status == ProbeStatus.FAIL
        assert finding.severity == Severity.HIGH
        assert any("多个不同的模型标识" in inf for inf in finding.inferences), finding.inferences
        assert analyze_risk([finding]) == RiskLevel.HIGH
        # 该 finding 引用的是真实证据
        assert set(finding.evidence_refs) == {str(e.evidence_id) for e in outcome.evidence}


class TestRawBytesHash:
    """9.5 哈希基于原始响应字节，response_body_size 为字节长度."""

    @pytest.mark.asyncio
    async def test_complete_hash_from_raw_bytes(self) -> None:
        """非流式响应含中文/换行/多字节字符时，哈希与大小基于原始字节."""
        payload = {
            "id": "chatcmpl-hash1",
            "object": "chat.completion",
            "created": 1,
            "model": "mock-model-v1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "你好\nLLMTrace"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        expected = hashlib.sha256(raw).hexdigest()

        with respx.mock as mock:
            mock.post("http://test.example.com/v1/chat/completions").respond(
                status_code=200,
                content=raw,
                headers={"Content-Type": "application/json"},
            )
            async with OpenAICompatibleProvider(_make_config(), API_KEY) as provider:
                evidence = await provider.complete("mock-model-v1", [{"role": "user", "content": "hi"}])

        assert evidence.response_body_sha256 == expected
        assert evidence.response_body_size == len(raw)
        assert "你好" in evidence.response_text

    @pytest.mark.asyncio
    async def test_stream_hash_from_raw_bytes(self) -> None:
        """流式哈希来自固定 SSE 原始字节，而不是解析后重新拼接的数据."""
        raw_sse = b'data: {"text":"A"}\n\ndata: {"text":"B"}\n\n'
        expected = hashlib.sha256(raw_sse).hexdigest()

        with respx.mock as mock:
            mock.post("http://test.example.com/v1/chat/completions").respond(
                status_code=200,
                content=raw_sse,
                headers={"Content-Type": "text/event-stream"},
            )
            async with OpenAICompatibleProvider(_make_config(), API_KEY) as provider:
                evidence = await provider.stream_complete("mock-model-v1", [{"role": "user", "content": "hi"}])

        assert evidence.response_body_sha256 == expected
        assert evidence.response_body_size == len(raw_sse)
        assert evidence.total_latency_ms is not None
        assert evidence.response_time is not None

    @pytest.mark.asyncio
    async def test_truncated_hash_still_full_response(self) -> None:
        """响应超限被截断摘要时，哈希仍对应完整原始字节."""
        payload = {
            "id": "chatcmpl-hash2",
            "object": "chat.completion",
            "created": 1,
            "model": "mock-model-v1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "x" * 500},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 500, "total_tokens": 501},
        }
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        expected = hashlib.sha256(raw).hexdigest()

        config = _make_config()
        config.max_response_bytes = 100

        with respx.mock as mock:
            mock.post("http://test.example.com/v1/chat/completions").respond(
                status_code=200,
                content=raw,
                headers={"Content-Type": "application/json"},
            )
            async with OpenAICompatibleProvider(config, API_KEY) as provider:
                evidence = await provider.complete("mock-model-v1", [{"role": "user", "content": "hi"}])

        assert evidence.response_truncated is True
        assert evidence.response_body_sha256 == expected
        assert evidence.response_body_size == len(raw)
        assert len(evidence.response_body_summary) <= config.max_response_bytes + 3  # 容忍半个多字节字符


class TestRequestIdExtraction:
    """request_id 大小写不敏感提取，response_id 独立保存."""

    def test_extract_request_id_case_insensitive(self) -> None:
        """不同大小写的请求 ID 响应头都能提取."""
        assert _extract_request_id({"Request-Id": "abc"}) == "abc"
        assert _extract_request_id({"X-REQUEST-ID": "def"}) == "def"
        assert _extract_request_id({"anthropic-request-id": "req-1"}) == "req-1"
        assert _extract_request_id({"OpenAI-Request-ID": "req-2"}) == "req-2"
        assert _extract_request_id({"x-amzn-requestid": "req-3"}) == "req-3"
        assert _extract_request_id({"Cf-Ray": "ray-1"}) == "ray-1"
        assert _extract_request_id({"X-Request-Id": "req-4"}) == "req-4"
        assert _extract_request_id({"other": "x"}) is None

    @pytest.mark.asyncio
    async def test_request_id_and_response_id_separated(self) -> None:
        """上游请求头 ID 与响应体 ID 分开保存."""
        payload = {
            "id": "chatcmpl-body-1",
            "object": "chat.completion",
            "created": 1,
            "model": "mock-model-v1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
        }

        with respx.mock as mock:
            mock.post("http://test.example.com/v1/chat/completions").respond(
                status_code=200,
                json=payload,
                headers={"X-REQUEST-ID": "upstream-header-1"},
            )
            async with OpenAICompatibleProvider(_make_config(), API_KEY) as provider:
                evidence = await provider.complete("mock-model-v1", [{"role": "user", "content": "hi"}])

        assert evidence.request_id == "upstream-header-1"
        assert evidence.response_id == "chatcmpl-body-1"
        assert evidence.request_id != evidence.response_id
