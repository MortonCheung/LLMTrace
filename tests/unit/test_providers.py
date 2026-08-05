"""Unit tests for LLMTrace providers."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from llmtrace.config import AuditConfig, AuthStyle, Protocol
from llmtrace.providers.anthropic_compatible import AnthropicCompatibleProvider
from llmtrace.providers.openai_compatible import OpenAICompatibleProvider
from llmtrace.providers.url_utils import join_url

API_KEY = "test-api-key-12345"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config_openai() -> AuditConfig:
    """OpenAI 协议审计配置."""
    return AuditConfig(
        protocol=Protocol.OPENAI,
        base_url="http://test.example.com",
        model="test-model",
        api_key_env="TEST_KEY",
        timeout=10.0,
        max_output_tokens=64,
    )


@pytest.fixture
def config_anthropic() -> AuditConfig:
    """Anthropic 协议审计配置."""
    return AuditConfig(
        protocol=Protocol.ANTHROPIC,
        base_url="http://test.example.com",
        model="test-model",
        api_key_env="TEST_KEY",
        timeout=10.0,
        max_output_tokens=64,
    )


# ---------------------------------------------------------------------------
# join_url tests
# ---------------------------------------------------------------------------


class TestJoinURL:
    """join_url 函数单元测试."""

    def test_basic_join(self) -> None:
        """简单拼接：base 无 /v1，path 无 /v1."""
        result = join_url("http://test.example.com", "models")
        assert result == "http://test.example.com/models"

    def test_base_with_v1_path_without(self) -> None:
        """base 以 /v1 结尾，path 不以 v1/ 开头."""
        result = join_url("http://test.example.com/v1", "models")
        assert result == "http://test.example.com/v1/models"

    def test_base_without_v1_path_with_v1(self) -> None:
        """base 不以 /v1 结尾，path 以 /v1/ 开头."""
        result = join_url("http://test.example.com", "/v1/models")
        assert result == "http://test.example.com/v1/models"

    def test_both_have_v1_dedup(self) -> None:
        """base 和 path 都有 /v1，应去重."""
        result = join_url("http://test.example.com/v1", "/v1/models")
        assert result == "http://test.example.com/v1/models"

    def test_both_have_v1_deeper_path(self) -> None:
        """base 和 path 都有 /v1，path 更深，应去重."""
        result = join_url("http://test.example.com/v1", "/v1/chat/completions")
        assert result == "http://test.example.com/v1/chat/completions"

    def test_base_trailing_slash(self) -> None:
        """base 有尾部斜杠."""
        result = join_url("http://test.example.com/", "models")
        assert result == "http://test.example.com/models"

    def test_path_leading_slash(self) -> None:
        """path 有前导斜杠."""
        result = join_url("http://test.example.com/api", "/models")
        assert result == "http://test.example.com/api/models"

    def test_both_slashes(self) -> None:
        """base 尾部斜杠 + path 前导斜杠."""
        result = join_url("http://test.example.com/", "/models")
        assert result == "http://test.example.com/models"

    def test_no_v1_anywhere(self) -> None:
        """base 和 path 都不含 /v1，正常拼接."""
        result = join_url("http://api.example.com/proxy", "/some/path")
        assert result == "http://api.example.com/proxy/some/path"

    def test_v1_in_middle_of_base(self) -> None:
        """v1 在 base 中间位置，不应影响拼接."""
        result = join_url("http://api.example.com/v1/proxy", "/v1/models")
        assert result == "http://api.example.com/v1/proxy/v1/models"

    def test_complex_base_with_v1(self) -> None:
        """复杂 base 含 /v1，path 含 /v1 更深路径."""
        result = join_url("http://api.example.com/proxy/v1", "/v1/chat/completions")
        assert result == "http://api.example.com/proxy/v1/chat/completions"


# ---------------------------------------------------------------------------
# OpenAICompatibleProvider tests
# ---------------------------------------------------------------------------


class TestOpenAICompatibleProvider:
    """OpenAICompatibleProvider 单元测试."""

    # -- list_models ---------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_models_success(self, config_openai: AuditConfig) -> None:
        """模型列表 API 调用成功，返回 200 及模型数据."""
        mock_data = {
            "object": "list",
            "data": [
                {"id": "gpt-4", "object": "model", "created": 1687882411},
                {"id": "gpt-3.5-turbo", "object": "model", "created": 1677649963},
            ],
        }

        with respx.mock as mock:
            mock.get("http://test.example.com/v1/models").respond(
                status_code=200,
                json=mock_data,
            )

            async with OpenAICompatibleProvider(config_openai, API_KEY) as provider:
                evidence, models = await provider.list_models()

        assert evidence.http_status == 200
        assert evidence.success is True
        assert models == ["gpt-4", "gpt-3.5-turbo"]
        assert evidence.request_method == "GET"

    @pytest.mark.asyncio
    async def test_list_models_empty_data(self, config_openai: AuditConfig) -> None:
        """模型列表 API 返回空 data 列表."""
        with respx.mock as mock:
            mock.get("http://test.example.com/v1/models").respond(
                status_code=200,
                json={"object": "list", "data": []},
            )

            async with OpenAICompatibleProvider(config_openai, API_KEY) as provider:
                evidence, models = await provider.list_models()

        assert evidence.http_status == 200
        assert models == []

    @pytest.mark.asyncio
    async def test_list_models_http_404(self, config_openai: AuditConfig) -> None:
        """模型列表 API 返回 404."""
        with respx.mock as mock:
            mock.get("http://test.example.com/v1/models").respond(
                status_code=404,
                json={"error": "Not Found"},
            )

            async with OpenAICompatibleProvider(config_openai, API_KEY) as provider:
                evidence, models = await provider.list_models()

        assert evidence.http_status == 404
        assert evidence.success is False
        assert models == []

    @pytest.mark.asyncio
    async def test_list_models_http_401(self, config_openai: AuditConfig) -> None:
        """模型列表 API 返回 401."""
        with respx.mock as mock:
            mock.get("http://test.example.com/v1/models").respond(
                status_code=401,
                json={"error": "Unauthorized"},
            )

            async with OpenAICompatibleProvider(config_openai, API_KEY) as provider:
                evidence, models = await provider.list_models()

        assert evidence.http_status == 401
        assert models == []

    @pytest.mark.asyncio
    async def test_list_models_http_500(self, config_openai: AuditConfig) -> None:
        """模型列表 API 返回 500."""
        with respx.mock as mock:
            mock.get("http://test.example.com/v1/models").respond(
                status_code=500,
                json={"error": "Internal Server Error"},
            )

            async with OpenAICompatibleProvider(config_openai, API_KEY) as provider:
                evidence, models = await provider.list_models()

        assert evidence.http_status == 500
        assert models == []

    # -- complete (non-streaming) --------------------------------------------

    @pytest.mark.asyncio
    async def test_complete_non_streaming(self, config_openai: AuditConfig) -> None:
        """非流式补全请求成功，返回完整响应."""
        mock_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello, how can I help you?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 7,
                "total_tokens": 17,
            },
        }

        with respx.mock as mock:
            mock.post("http://test.example.com/v1/chat/completions").respond(
                status_code=200,
                json=mock_response,
            )

            async with OpenAICompatibleProvider(config_openai, API_KEY) as provider:
                evidence = await provider.complete(
                    "gpt-4",
                    [{"role": "user", "content": "Hi"}],
                )

        assert evidence.http_status == 200
        assert evidence.success is True
        assert evidence.response_model == "gpt-4"
        assert evidence.response_id == "chatcmpl-123"
        assert evidence.response_text == "Hello, how can I help you?"
        assert evidence.finish_reason == "stop"
        assert evidence.input_tokens == 10
        assert evidence.output_tokens == 7
        assert evidence.request_method == "POST"

    @pytest.mark.asyncio
    async def test_complete_http_404(self, config_openai: AuditConfig) -> None:
        """非流式补全请求返回 404."""
        with respx.mock as mock:
            mock.post("http://test.example.com/v1/chat/completions").respond(
                status_code=404,
                json={"error": "Not Found"},
            )

            async with OpenAICompatibleProvider(config_openai, API_KEY) as provider:
                evidence = await provider.complete(
                    "gpt-4",
                    [{"role": "user", "content": "Hi"}],
                )

        assert evidence.http_status == 404
        assert evidence.success is False

    @pytest.mark.asyncio
    async def test_complete_http_401(self, config_openai: AuditConfig) -> None:
        """非流式补全请求返回 401."""
        with respx.mock as mock:
            mock.post("http://test.example.com/v1/chat/completions").respond(
                status_code=401,
                json={"error": "Unauthorized"},
            )

            async with OpenAICompatibleProvider(config_openai, API_KEY) as provider:
                evidence = await provider.complete(
                    "gpt-4",
                    [{"role": "user", "content": "Hi"}],
                )

        assert evidence.http_status == 401
        assert evidence.success is False

    @pytest.mark.asyncio
    async def test_complete_http_500(self, config_openai: AuditConfig) -> None:
        """非流式补全请求返回 500."""
        with respx.mock as mock:
            mock.post("http://test.example.com/v1/chat/completions").respond(
                status_code=500,
                json={"error": "Internal Server Error"},
            )

            async with OpenAICompatibleProvider(config_openai, API_KEY) as provider:
                evidence = await provider.complete(
                    "gpt-4",
                    [{"role": "user", "content": "Hi"}],
                )

        assert evidence.http_status == 500
        assert evidence.success is False

    @pytest.mark.asyncio
    async def test_complete_json_parse_error(self, config_openai: AuditConfig) -> None:
        """非流式补全请求返回 200 但 body 为无效 JSON."""
        with respx.mock as mock:
            mock.post("http://test.example.com/v1/chat/completions").respond(
                status_code=200,
                content=b"not valid json",
                headers={"Content-Type": "application/json"},
            )

            async with OpenAICompatibleProvider(config_openai, API_KEY) as provider:
                evidence = await provider.complete(
                    "gpt-4",
                    [{"role": "user", "content": "Hi"}],
                )

        assert evidence.http_status == 200
        assert evidence.exception_type is not None

    @pytest.mark.asyncio
    async def test_complete_timeout(self, config_openai: AuditConfig) -> None:
        """非流式补全请求超时."""
        with respx.mock as mock:
            mock.post("http://test.example.com/v1/chat/completions").mock(
                side_effect=httpx.TimeoutException("Connection timed out"),
            )

            async with OpenAICompatibleProvider(config_openai, API_KEY) as provider:
                evidence = await provider.complete(
                    "gpt-4",
                    [{"role": "user", "content": "Hi"}],
                )

        assert evidence.exception_type == "TimeoutException"
        assert evidence.exception_message == "Connection timed out"

    # -- stream_complete -----------------------------------------------------

    @pytest.mark.asyncio
    async def test_stream_complete(self, config_openai: AuditConfig) -> None:
        """流式补全请求成功，解析 SSE 事件."""
        sse_data = (
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
            '"created":1694268190,"model":"gpt-4","choices":[{"index":0,'
            '"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n'
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
            '"created":1694268190,"model":"gpt-4","choices":[{"index":0,'
            '"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
            '"created":1694268190,"model":"gpt-4","choices":[{"index":0,'
            '"delta":{"content":" world"},"finish_reason":null}]}\n\n'
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
            '"created":1694268190,"model":"gpt-4","choices":[{"index":0,'
            '"delta":{"content":"!"},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}}\n\n'
            "data: [DONE]\n\n"
        )

        with respx.mock as mock:
            mock.post("http://test.example.com/v1/chat/completions").respond(
                status_code=200,
                content=sse_data,
                headers={"Content-Type": "text/event-stream"},
            )

            async with OpenAICompatibleProvider(config_openai, API_KEY) as provider:
                evidence = await provider.stream_complete(
                    "gpt-4",
                    [{"role": "user", "content": "Hi"}],
                )

        assert evidence.http_status == 200
        assert evidence.response_text == "Hello world!"
        assert evidence.finish_reason == "stop"
        assert evidence.input_tokens == 10
        assert evidence.output_tokens == 3
        assert evidence.response_model == "gpt-4"
        assert evidence.response_id == "chatcmpl-123"
        assert evidence.first_token_latency_ms is not None

    @pytest.mark.asyncio
    async def test_stream_complete_empty_events(self, config_openai: AuditConfig) -> None:
        """流式补全请求仅含 [DONE] 事件，无有效内容."""
        sse_data = "data: [DONE]\n\n"

        with respx.mock as mock:
            mock.post("http://test.example.com/v1/chat/completions").respond(
                status_code=200,
                content=sse_data,
                headers={"Content-Type": "text/event-stream"},
            )

            async with OpenAICompatibleProvider(config_openai, API_KEY) as provider:
                evidence = await provider.stream_complete(
                    "gpt-4",
                    [{"role": "user", "content": "Hi"}],
                )

        assert evidence.http_status == 200
        assert evidence.response_text == ""

    @pytest.mark.asyncio
    async def test_stream_complete_http_401(self, config_openai: AuditConfig) -> None:
        """流式补全请求返回 401."""
        with respx.mock as mock:
            mock.post("http://test.example.com/v1/chat/completions").respond(
                status_code=401,
                json={"error": "Unauthorized"},
            )

            async with OpenAICompatibleProvider(config_openai, API_KEY) as provider:
                evidence = await provider.stream_complete(
                    "gpt-4",
                    [{"role": "user", "content": "Hi"}],
                )

        assert evidence.http_status == 401
        assert evidence.success is False

    # -- headers -------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_build_headers(self, config_openai: AuditConfig) -> None:
        """验证 Bearer Authorization 请求头."""
        with respx.mock as mock:
            route = mock.get("http://test.example.com/v1/models").respond(
                status_code=200,
                json={"object": "list", "data": []},
            )

            async with OpenAICompatibleProvider(config_openai, API_KEY) as provider:
                await provider.list_models()

            request = route.calls[0].request
            assert request.headers["Authorization"] == f"Bearer {API_KEY}"
            assert request.headers["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# AnthropicCompatibleProvider tests
# ---------------------------------------------------------------------------


class TestAnthropicCompatibleProvider:
    """AnthropicCompatibleProvider 单元测试."""

    # -- list_models ---------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_models_success(self, config_anthropic: AuditConfig) -> None:
        """模型列表 API 调用成功，返回 200 及模型数据."""
        mock_data = {
            "data": [
                {"id": "claude-3-opus-20240229", "object": "model", "created": 1708560000},
                {"id": "claude-3-sonnet-20240229", "object": "model", "created": 1708560000},
            ],
        }

        with respx.mock as mock:
            mock.get("http://test.example.com/v1/models").respond(
                status_code=200,
                json=mock_data,
            )

            async with AnthropicCompatibleProvider(config_anthropic, API_KEY) as provider:
                evidence, models = await provider.list_models()

        assert evidence.http_status == 200
        assert evidence.success is True
        assert models == ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]

    @pytest.mark.asyncio
    async def test_list_models_http_500(self, config_anthropic: AuditConfig) -> None:
        """模型列表 API 返回 500."""
        with respx.mock as mock:
            mock.get("http://test.example.com/v1/models").respond(
                status_code=500,
                json={"error": "Server Error"},
            )

            async with AnthropicCompatibleProvider(config_anthropic, API_KEY) as provider:
                evidence, models = await provider.list_models()

        assert evidence.http_status == 500
        assert models == []

    # -- complete (non-streaming) --------------------------------------------

    @pytest.mark.asyncio
    async def test_complete_non_streaming(self, config_anthropic: AuditConfig) -> None:
        """非流式 Messages API 调用成功，返回完整响应."""
        mock_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-opus-20240229",
            "content": [{"type": "text", "text": "Hello! How can I assist you today?"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 8,
            },
        }

        with respx.mock as mock:
            mock.post("http://test.example.com/v1/messages").respond(
                status_code=200,
                json=mock_response,
            )

            async with AnthropicCompatibleProvider(config_anthropic, API_KEY) as provider:
                evidence = await provider.complete(
                    "claude-3-opus-20240229",
                    [{"role": "user", "content": "Hello"}],
                )

        assert evidence.http_status == 200
        assert evidence.success is True
        assert evidence.response_model == "claude-3-opus-20240229"
        assert evidence.response_id == "msg_123"
        assert evidence.response_text == "Hello! How can I assist you today?"
        assert evidence.finish_reason == "end_turn"
        assert evidence.input_tokens == 10
        assert evidence.output_tokens == 8

    @pytest.mark.asyncio
    async def test_complete_with_system_prompt(self, config_anthropic: AuditConfig) -> None:
        """包含 system prompt 的补全请求，验证 system 字段独立于 messages."""
        mock_response = {
            "id": "msg_456",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-opus-20240229",
            "content": [{"type": "text", "text": "I understand."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 20, "output_tokens": 3},
        }

        with respx.mock as mock:
            route = mock.post("http://test.example.com/v1/messages").respond(
                status_code=200,
                json=mock_response,
            )

            async with AnthropicCompatibleProvider(config_anthropic, API_KEY) as provider:
                await provider.complete(
                    "claude-3-opus-20240229",
                    [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Hello"},
                    ],
                )

        request_body = json.loads(route.calls[0].request.content)
        assert request_body["system"] == "You are helpful."
        assert len(request_body["messages"]) == 1

    @pytest.mark.asyncio
    async def test_complete_http_404(self, config_anthropic: AuditConfig) -> None:
        """非流式 Messages API 返回 404."""
        with respx.mock as mock:
            mock.post("http://test.example.com/v1/messages").respond(
                status_code=404,
                json={"error": "Not Found"},
            )

            async with AnthropicCompatibleProvider(config_anthropic, API_KEY) as provider:
                evidence = await provider.complete(
                    "claude-3-opus-20240229",
                    [{"role": "user", "content": "Hello"}],
                )

        assert evidence.http_status == 404
        assert evidence.success is False

    @pytest.mark.asyncio
    async def test_complete_http_401(self, config_anthropic: AuditConfig) -> None:
        """非流式 Messages API 返回 401."""
        with respx.mock as mock:
            mock.post("http://test.example.com/v1/messages").respond(
                status_code=401,
                json={"error": "Unauthorized"},
            )

            async with AnthropicCompatibleProvider(config_anthropic, API_KEY) as provider:
                evidence = await provider.complete(
                    "claude-3-opus-20240229",
                    [{"role": "user", "content": "Hello"}],
                )

        assert evidence.http_status == 401
        assert evidence.success is False

    @pytest.mark.asyncio
    async def test_complete_http_500(self, config_anthropic: AuditConfig) -> None:
        """非流式 Messages API 返回 500."""
        with respx.mock as mock:
            mock.post("http://test.example.com/v1/messages").respond(
                status_code=500,
                json={"error": "Internal Server Error"},
            )

            async with AnthropicCompatibleProvider(config_anthropic, API_KEY) as provider:
                evidence = await provider.complete(
                    "claude-3-opus-20240229",
                    [{"role": "user", "content": "Hello"}],
                )

        assert evidence.http_status == 500
        assert evidence.success is False

    # -- stream_complete -----------------------------------------------------

    @pytest.mark.asyncio
    async def test_stream_complete(self, config_anthropic: AuditConfig) -> None:
        """流式 Messages API 调用成功，解析 Anthropic SSE 事件."""
        sse_data = (
            'data: {"type":"message_start","message":{"id":"msg_123","type":"message",'
            '"role":"assistant","model":"claude-3-opus-20240229","content":[],'
            '"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}\n\n'
            'data: {"type":"content_block_start","index":0,'
            '"content_block":{"type":"text","text":""}}\n\n'
            'data: {"type":"content_block_delta","index":0,'
            '"delta":{"type":"text_delta","text":"Hello"}}\n\n'
            'data: {"type":"content_block_delta","index":0,'
            '"delta":{"type":"text_delta","text":" world"}}\n\n'
            'data: {"type":"content_block_delta","index":0,'
            '"delta":{"type":"text_delta","text":"!"}}\n\n'
            'data: {"type":"content_block_stop","index":0}\n\n'
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn",'
            '"stop_sequence":null},"usage":{"output_tokens":5}}\n\n'
            'data: {"type":"message_stop"}\n\n'
        )

        with respx.mock as mock:
            mock.post("http://test.example.com/v1/messages").respond(
                status_code=200,
                content=sse_data,
                headers={"Content-Type": "text/event-stream"},
            )

            async with AnthropicCompatibleProvider(config_anthropic, API_KEY) as provider:
                evidence = await provider.stream_complete(
                    "claude-3-opus-20240229",
                    [{"role": "user", "content": "Hello"}],
                )

        assert evidence.http_status == 200
        assert evidence.response_text == "Hello world!"
        assert evidence.finish_reason == "end_turn"
        assert evidence.output_tokens == 5
        assert evidence.response_model == "claude-3-opus-20240229"
        assert evidence.response_id == "msg_123"
        assert evidence.first_token_latency_ms is not None

    @pytest.mark.asyncio
    async def test_stream_complete_http_401(self, config_anthropic: AuditConfig) -> None:
        """流式 Messages API 返回 401."""
        with respx.mock as mock:
            mock.post("http://test.example.com/v1/messages").respond(
                status_code=401,
                json={"error": "Unauthorized"},
            )

            async with AnthropicCompatibleProvider(config_anthropic, API_KEY) as provider:
                evidence = await provider.stream_complete(
                    "claude-3-opus-20240229",
                    [{"role": "user", "content": "Hello"}],
                )

        assert evidence.http_status == 401
        assert evidence.success is False

    # -- auth style ----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_auth_style_default_auto(self) -> None:
        """auth_style=auto 默认使用 x-api-key header."""
        config = AuditConfig(
            protocol=Protocol.ANTHROPIC,
            base_url="http://test.example.com",
            model="test-model",
            api_key_env="TEST_KEY",
            auth_style=AuthStyle.AUTO,
            timeout=10.0,
            max_output_tokens=64,
        )

        with respx.mock as mock:
            route = mock.get("http://test.example.com/v1/models").respond(
                status_code=200,
                json={"data": []},
            )

            async with AnthropicCompatibleProvider(config, API_KEY) as provider:
                await provider.list_models()

            request = route.calls[0].request
            assert request.headers.get("x-api-key") == API_KEY
            assert "Authorization" not in request.headers

    @pytest.mark.asyncio
    async def test_auth_style_bearer(self) -> None:
        """auth_style=bearer 仅使用 Authorization header."""
        config = AuditConfig(
            protocol=Protocol.ANTHROPIC,
            base_url="http://test.example.com",
            model="test-model",
            api_key_env="TEST_KEY",
            auth_style=AuthStyle.BEARER,
            timeout=10.0,
            max_output_tokens=64,
        )

        with respx.mock as mock:
            route = mock.get("http://test.example.com/v1/models").respond(
                status_code=200,
                json={"data": []},
            )

            async with AnthropicCompatibleProvider(config, API_KEY) as provider:
                await provider.list_models()

            request = route.calls[0].request
            assert request.headers.get("Authorization") == f"Bearer {API_KEY}"
            assert "x-api-key" not in request.headers

    @pytest.mark.asyncio
    async def test_auth_style_x_api_key(self) -> None:
        """auth_style=x-api-key 仅使用 x-api-key header."""
        config = AuditConfig(
            protocol=Protocol.ANTHROPIC,
            base_url="http://test.example.com",
            model="test-model",
            api_key_env="TEST_KEY",
            auth_style=AuthStyle.X_API_KEY,
            timeout=10.0,
            max_output_tokens=64,
        )

        with respx.mock as mock:
            route = mock.get("http://test.example.com/v1/models").respond(
                status_code=200,
                json={"data": []},
            )

            async with AnthropicCompatibleProvider(config, API_KEY) as provider:
                await provider.list_models()

            request = route.calls[0].request
            assert request.headers.get("x-api-key") == API_KEY
            assert "Authorization" not in request.headers

    @pytest.mark.asyncio
    async def test_auth_style_both(self) -> None:
        """auth_style=both 同时使用两种 header."""
        config = AuditConfig(
            protocol=Protocol.ANTHROPIC,
            base_url="http://test.example.com",
            model="test-model",
            api_key_env="TEST_KEY",
            auth_style=AuthStyle.BOTH,
            timeout=10.0,
            max_output_tokens=64,
        )

        with respx.mock as mock:
            route = mock.get("http://test.example.com/v1/models").respond(
                status_code=200,
                json={"data": []},
            )

            async with AnthropicCompatibleProvider(config, API_KEY) as provider:
                await provider.list_models()

            request = route.calls[0].request
            assert request.headers.get("Authorization") == f"Bearer {API_KEY}"
            assert request.headers.get("x-api-key") == API_KEY

    # -- anthropic-version header --------------------------------------------

    @pytest.mark.asyncio
    async def test_anthropic_version_header(self, config_anthropic: AuditConfig) -> None:
        """验证 anthropic-version 请求头."""
        with respx.mock as mock:
            route = mock.get("http://test.example.com/v1/models").respond(
                status_code=200,
                json={"data": []},
            )

            async with AnthropicCompatibleProvider(config_anthropic, API_KEY) as provider:
                await provider.list_models()

            request = route.calls[0].request
            assert request.headers.get("anthropic-version") == "2023-06-01"


# ---------------------------------------------------------------------------
# Provider CompletionOptions mapping tests (respx)
# ---------------------------------------------------------------------------


class TestOpenAICompletionOptionsMapping:
    """Verifies CompletionOptions are mapped into the OpenAI request body."""

    @pytest.mark.asyncio
    async def test_maps_temperature(self, config_openai: AuditConfig) -> None:
        from llmtrace.benchmarks.models import CompletionOptions

        options = CompletionOptions(temperature=0.3)

        with respx.mock as mock:
            route = mock.post("http://test.example.com/v1/chat/completions").respond(
                status_code=200,
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            )

            async with OpenAICompatibleProvider(config_openai, API_KEY) as provider:
                await provider.complete("gpt-4", [{"role": "user", "content": "Hi"}], options=options)

            body = json.loads(route.calls[0].request.content)
            assert body["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_maps_stop_sequences(self, config_openai: AuditConfig) -> None:
        from llmtrace.benchmarks.models import CompletionOptions

        options = CompletionOptions(stop=["END", "STOP"])

        with respx.mock as mock:
            route = mock.post("http://test.example.com/v1/chat/completions").respond(
                status_code=200,
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            )

            async with OpenAICompatibleProvider(config_openai, API_KEY) as provider:
                await provider.complete("gpt-4", [{"role": "user", "content": "Hi"}], options=options)

            body = json.loads(route.calls[0].request.content)
            assert body["stop"] == ["END", "STOP"]

    @pytest.mark.asyncio
    async def test_maps_until_as_stop(self, config_openai: AuditConfig) -> None:
        from llmtrace.benchmarks.models import CompletionOptions

        options = CompletionOptions(until=["\n"])

        with respx.mock as mock:
            route = mock.post("http://test.example.com/v1/chat/completions").respond(
                status_code=200,
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            )

            async with OpenAICompatibleProvider(config_openai, API_KEY) as provider:
                await provider.complete("gpt-4", [{"role": "user", "content": "Hi"}], options=options)

            body = json.loads(route.calls[0].request.content)
            assert body["stop"] == ["\n"]

    @pytest.mark.asyncio
    async def test_maps_max_tokens(self, config_openai: AuditConfig) -> None:
        from llmtrace.benchmarks.models import CompletionOptions

        options = CompletionOptions(max_tokens=256)

        with respx.mock as mock:
            route = mock.post("http://test.example.com/v1/chat/completions").respond(
                status_code=200,
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            )

            async with OpenAICompatibleProvider(config_openai, API_KEY) as provider:
                await provider.complete("gpt-4", [{"role": "user", "content": "Hi"}], options=options)

            body = json.loads(route.calls[0].request.content)
            assert body["max_tokens"] == 256

    @pytest.mark.asyncio
    async def test_maps_max_gen_toks_as_max_tokens(self, config_openai: AuditConfig) -> None:
        from llmtrace.benchmarks.models import CompletionOptions

        options = CompletionOptions(max_gen_toks=128)

        with respx.mock as mock:
            route = mock.post("http://test.example.com/v1/chat/completions").respond(
                status_code=200,
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            )

            async with OpenAICompatibleProvider(config_openai, API_KEY) as provider:
                await provider.complete("gpt-4", [{"role": "user", "content": "Hi"}], options=options)

            body = json.loads(route.calls[0].request.content)
            assert body["max_tokens"] == 128

    @pytest.mark.asyncio
    async def test_conflicting_max_tokens_raises(self, config_openai: AuditConfig) -> None:
        from llmtrace.benchmarks.models import CompletionOptions

        options = CompletionOptions(max_tokens=100, max_gen_toks=200)

        async with OpenAICompatibleProvider(config_openai, API_KEY) as provider:
            with pytest.raises(ValueError, match="Conflicting token limits"):
                await provider.complete("gpt-4", [{"role": "user", "content": "Hi"}], options=options)

    @pytest.mark.asyncio
    async def test_do_sample_false_accepted(self, config_openai: AuditConfig) -> None:
        from llmtrace.benchmarks.models import CompletionOptions

        options = CompletionOptions(do_sample=False)

        with respx.mock as mock:
            mock.post("http://test.example.com/v1/chat/completions").respond(
                status_code=200,
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            )

            async with OpenAICompatibleProvider(config_openai, API_KEY) as provider:
                evidence = await provider.complete("gpt-4", [{"role": "user", "content": "Hi"}], options=options)

            assert evidence.http_status == 200

    @pytest.mark.asyncio
    async def test_do_sample_true_raises(self, config_openai: AuditConfig) -> None:
        from llmtrace.benchmarks.models import CompletionOptions

        options = CompletionOptions(do_sample=True)

        async with OpenAICompatibleProvider(config_openai, API_KEY) as provider:
            with pytest.raises(ValueError, match="do_sample=True is not supported"):
                await provider.complete("gpt-4", [{"role": "user", "content": "Hi"}], options=options)

    @pytest.mark.asyncio
    async def test_combines_stop_and_until(self, config_openai: AuditConfig) -> None:
        from llmtrace.benchmarks.models import CompletionOptions

        options = CompletionOptions(stop=["END"], until=["\n", "END"])

        with respx.mock as mock:
            route = mock.post("http://test.example.com/v1/chat/completions").respond(
                status_code=200,
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            )

            async with OpenAICompatibleProvider(config_openai, API_KEY) as provider:
                await provider.complete("gpt-4", [{"role": "user", "content": "Hi"}], options=options)

            body = json.loads(route.calls[0].request.content)
            assert body["stop"] == ["END", "\n"]  # deduplicated: END not repeated

    @pytest.mark.asyncio
    async def test_no_options_unchanged_body(self, config_openai: AuditConfig) -> None:
        with respx.mock as mock:
            route = mock.post("http://test.example.com/v1/chat/completions").respond(
                status_code=200,
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            )

            async with OpenAICompatibleProvider(config_openai, API_KEY) as provider:
                await provider.complete("gpt-4", [{"role": "user", "content": "Hi"}])

            body = json.loads(route.calls[0].request.content)
            assert "stop" not in body  # no stop without options
            assert "temperature" not in body


class TestAnthropicCompletionOptionsMapping:
    """Verifies CompletionOptions are mapped into the Anthropic request body."""

    @pytest.mark.asyncio
    async def test_maps_temperature(self, config_anthropic: AuditConfig) -> None:
        from llmtrace.benchmarks.models import CompletionOptions

        options = CompletionOptions(temperature=0.5)

        with respx.mock as mock:
            route = mock.post("http://test.example.com/v1/messages").respond(
                status_code=200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3-opus-20240229",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            )

            async with AnthropicCompatibleProvider(config_anthropic, API_KEY) as provider:
                await provider.complete("claude-3-opus-20240229", [{"role": "user", "content": "Hi"}], options=options)

            body = json.loads(route.calls[0].request.content)
            assert body["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_maps_stop_to_stop_sequences(self, config_anthropic: AuditConfig) -> None:
        from llmtrace.benchmarks.models import CompletionOptions

        options = CompletionOptions(stop=["END"])

        with respx.mock as mock:
            route = mock.post("http://test.example.com/v1/messages").respond(
                status_code=200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3-opus-20240229",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            )

            async with AnthropicCompatibleProvider(config_anthropic, API_KEY) as provider:
                await provider.complete("claude-3-opus-20240229", [{"role": "user", "content": "Hi"}], options=options)

            body = json.loads(route.calls[0].request.content)
            assert body["stop_sequences"] == ["END"]

    @pytest.mark.asyncio
    async def test_maps_max_tokens(self, config_anthropic: AuditConfig) -> None:
        from llmtrace.benchmarks.models import CompletionOptions

        options = CompletionOptions(max_tokens=512)

        with respx.mock as mock:
            route = mock.post("http://test.example.com/v1/messages").respond(
                status_code=200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3-opus-20240229",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            )

            async with AnthropicCompatibleProvider(config_anthropic, API_KEY) as provider:
                await provider.complete("claude-3-opus-20240229", [{"role": "user", "content": "Hi"}], options=options)

            body = json.loads(route.calls[0].request.content)
            assert body["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_do_sample_false_accepted(self, config_anthropic: AuditConfig) -> None:
        from llmtrace.benchmarks.models import CompletionOptions

        options = CompletionOptions(do_sample=False)

        with respx.mock as mock:
            mock.post("http://test.example.com/v1/messages").respond(
                status_code=200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3-opus-20240229",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            )

            async with AnthropicCompatibleProvider(config_anthropic, API_KEY) as provider:
                evidence = await provider.complete(
                    "claude-3-opus-20240229", [{"role": "user", "content": "Hi"}], options=options
                )

            assert evidence.http_status == 200

    @pytest.mark.asyncio
    async def test_do_sample_true_raises(self, config_anthropic: AuditConfig) -> None:
        from llmtrace.benchmarks.models import CompletionOptions

        options = CompletionOptions(do_sample=True)

        async with AnthropicCompatibleProvider(config_anthropic, API_KEY) as provider:
            with pytest.raises(ValueError, match="do_sample=True is not supported"):
                await provider.complete("claude-3-opus-20240229", [{"role": "user", "content": "Hi"}], options=options)
