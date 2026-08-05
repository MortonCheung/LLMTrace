"""Provider 基类."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime

import httpx

from llmtrace.config import AuditConfig
from llmtrace.models.evidence import HTTPEvidence
from llmtrace.security.redaction import redact_headers, redact_json_body, redact_url
from llmtrace.utilities.hashing import sha256_hash

# 需要从响应头提取 request_id 的 header 名称（大小写不敏感）
_REQUEST_ID_HEADERS = [
    "request-id",
    "x-request-id",
    "anthropic-request-id",
    "openai-request-id",
    "x-amzn-requestid",
    "cf-ray",
]


def _extract_request_id(headers: dict[str, str]) -> str | None:
    """从响应头中提取上游请求 ID（大小写不敏感）."""
    lower_headers = {k.lower(): v for k, v in headers.items()}
    for name in _REQUEST_ID_HEADERS:
        if name in lower_headers:
            return lower_headers[name]
    return None


class BaseProvider(ABC):
    """Provider 抽象基类."""

    def __init__(self, config: AuditConfig, api_key: str) -> None:
        self.config = config
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """获取 HTTP 客户端."""
        if self._client is None:
            raise RuntimeError("Provider not initialized. Call `async with provider:` first.")
        return self._client

    async def __aenter__(self) -> BaseProvider:
        self._client = httpx.AsyncClient(timeout=self.config.timeout)
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @abstractmethod
    def _build_headers(self) -> dict[str, str]:
        """构建请求头."""
        ...

    @abstractmethod
    def _build_models_url(self) -> str:
        """构建模型列表 URL."""
        ...

    @abstractmethod
    def _build_completion_url(self) -> str:
        """构建补全请求 URL."""
        ...

    @abstractmethod
    def _build_completion_body(self, model: str, messages: list[dict[str, str]]) -> dict[str, object]:
        """构建补全请求体."""
        ...

    @abstractmethod
    def _build_stream_body(self, model: str, messages: list[dict[str, str]]) -> dict[str, object]:
        """构建流式请求体."""
        ...

    @abstractmethod
    def _parse_response(self, data: dict[str, object], evidence: HTTPEvidence) -> None:
        """解析响应数据到证据."""
        ...

    @abstractmethod
    def _parse_stream_event(self, line: str) -> dict[str, object] | None:
        """解析流式事件."""
        ...

    async def list_models(self) -> tuple[HTTPEvidence, list[str]]:
        """获取模型列表."""
        evidence = self._build_evidence("GET", self._build_models_url())
        evidence.evidence_type = "model_catalog"
        url = self._build_models_url()
        headers = self._build_headers()

        try:
            evidence.request_time = datetime.now(tz=UTC)
            start = time.monotonic()
            response = await self.client.get(url, headers=headers)
            elapsed = (time.monotonic() - start) * 1000
            evidence.total_latency_ms = elapsed
            evidence.response_time = datetime.now(tz=UTC)

            evidence.http_status = response.status_code
            evidence.response_headers = dict(response.headers)
            evidence.request_id = _extract_request_id(evidence.response_headers)
            full_bytes = response.content
            evidence.response_body_size = len(full_bytes)
            evidence.response_body_sha256 = sha256_hash(full_bytes)

            if response.status_code == 200:
                data = response.json()
                models = self._extract_models(data)
                evidence.response_body_summary = full_bytes.decode("utf-8", errors="replace")[:2000]
                return evidence, models
            else:
                evidence.response_body_summary = full_bytes.decode("utf-8", errors="replace")[:2000]
                return evidence, []

        except Exception as e:
            evidence.exception_type = type(e).__name__
            evidence.exception_message = str(e)
            return evidence, []

    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        options: object | None = None,
    ) -> HTTPEvidence:
        """非流式补全请求.

        Args:
            model: Model identifier.
            messages: Chat messages.
            options: Optional CompletionOptions for generation kwargs.
                     Currently reserved for future use; not yet applied to the request.
        """
        # options are accepted but not yet applied to the request body.
        # Real providers can override to map them into provider-specific formats.
        _ = options

        url = self._build_completion_url()
        headers = self._build_headers()
        body = self._build_completion_body(model, messages)
        evidence = self._build_evidence("POST", url, body, model=model)

        try:
            evidence.request_time = datetime.now(tz=UTC)
            start = time.monotonic()
            response = await self.client.post(url, headers=headers, json=body)
            elapsed = (time.monotonic() - start) * 1000
            evidence.total_latency_ms = elapsed
            evidence.response_time = datetime.now(tz=UTC)

            evidence.http_status = response.status_code
            evidence.response_headers = dict(response.headers)
            evidence.request_id = _extract_request_id(evidence.response_headers)

            # 响应体按原始字节处理：哈希基于完整字节，摘要再按字节截断
            full_bytes = response.content
            evidence.response_body_size = len(full_bytes)
            evidence.response_body_sha256 = sha256_hash(full_bytes)

            if len(full_bytes) > self.config.max_response_bytes:
                evidence.response_truncated = True
                evidence.response_body_summary = full_bytes[: self.config.max_response_bytes].decode(
                    "utf-8", errors="replace"
                )
            else:
                evidence.response_body_summary = full_bytes.decode("utf-8", errors="replace")[:2000]

            if response.status_code == 200:
                data = response.json()
                self._parse_response(data, evidence)

        except Exception as e:
            evidence.exception_type = type(e).__name__
            evidence.exception_message = str(e)

        return evidence

    async def stream_complete(self, model: str, messages: list[dict[str, str]]) -> HTTPEvidence:
        """流式补全请求."""
        url = self._build_completion_url()
        headers = self._build_headers()
        body = self._build_stream_body(model, messages)
        evidence = self._build_evidence("POST", url, body, model=model)

        evidence.request_time = datetime.now(tz=UTC)
        start = time.monotonic()
        first_token_time: float | None = None
        full_text_parts: list[str] = []
        raw_bytes = bytearray()
        line_buffer = b""

        try:
            async with self.client.stream("POST", url, headers=headers, json=body) as response:
                evidence.http_status = response.status_code
                evidence.response_headers = dict(response.headers)
                evidence.request_id = _extract_request_id(evidence.response_headers)

                # 按原始字节累积，同时通过增量行缓冲解析 SSE 事件
                async for chunk in response.aiter_bytes():
                    raw_bytes.extend(chunk)
                    line_buffer += chunk
                    while b"\n" in line_buffer:
                        line_bytes, line_buffer = line_buffer.split(b"\n", 1)
                        line = line_bytes.decode("utf-8", errors="replace").rstrip("\r")
                        parsed = self._parse_stream_event(line)
                        if parsed:
                            text = self._extract_stream_text(parsed)
                            # 首 Token 延迟：从第一段非空文本 delta 计算
                            if first_token_time is None and text:
                                first_token_time = time.monotonic()
                                evidence.first_token_latency_ms = (first_token_time - start) * 1000
                            if text:
                                full_text_parts.append(text)
                            self._parse_stream_finish(parsed, evidence)

                stream_end = time.monotonic()
                evidence.total_latency_ms = (stream_end - start) * 1000
                evidence.response_time = datetime.now(tz=UTC)

        except Exception as e:
            evidence.exception_type = type(e).__name__
            evidence.exception_message = str(e)

        # 无论是否异常，只要有已采集的原始字节就保留证据
        if raw_bytes:
            evidence.response_text = "".join(full_text_parts)
            full_bytes = bytes(raw_bytes)
            evidence.response_body_size = len(full_bytes)
            # 哈希基于完整原始字节，摘要再按字节截断
            evidence.response_body_sha256 = sha256_hash(full_bytes)

            if len(full_bytes) > self.config.max_response_bytes:
                evidence.response_truncated = True
                evidence.response_body_summary = full_bytes[: self.config.max_response_bytes].decode(
                    "utf-8", errors="replace"
                )
            else:
                evidence.response_body_summary = full_bytes.decode("utf-8", errors="replace")[:2000]

            # 如果异常导致未设置结束时间，使用当前时间
            if evidence.response_time is None:
                evidence.response_time = datetime.now(tz=UTC)
            if evidence.total_latency_ms is None:
                evidence.total_latency_ms = (time.monotonic() - start) * 1000

        return evidence

    def _build_evidence(
        self,
        method: str,
        url: str,
        body: dict[str, object] | None = None,
        model: str | None = None,
    ) -> HTTPEvidence:
        """构建基础证据对象."""
        return HTTPEvidence(
            request_method=method,
            request_url_redacted=redact_url(url),
            request_path=url.replace(self.config.base_url, ""),
            request_headers_redacted=redact_headers(self._build_headers()),
            request_body_redacted=redact_json_body(body),
            request_model=model if model is not None else self.config.model,
        )

    @abstractmethod
    def _extract_models(self, data: dict[str, object]) -> list[str]:
        """从模型列表响应中提取模型 ID."""
        ...

    @abstractmethod
    def _extract_stream_text(self, event: dict[str, object]) -> str | None:
        """从流式事件中提取文本."""
        ...

    @abstractmethod
    def _parse_stream_finish(self, event: dict[str, object], evidence: HTTPEvidence) -> None:
        """从流式事件中提取完成信息."""
        ...
