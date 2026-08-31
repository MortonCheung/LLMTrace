"""Anthropic-compatible Provider."""

from __future__ import annotations

import json
from typing import Any

from llmtrace.benchmarks.models import CompletionOptions
from llmtrace.config import AuthStyle
from llmtrace.models.evidence import HTTPEvidence
from llmtrace.providers.base import (
    BaseProvider,
    _resolve_max_tokens,
    _resolve_stop_sequences,
    _validate_do_sample,
)
from llmtrace.providers.url_utils import join_url

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicCompatibleProvider(BaseProvider):
    """Anthropic-compatible 协议 Provider."""

    def _resolve_auth_style(self) -> AuthStyle:
        """解析鉴权方式."""
        style = self.config.auth_style
        if style == AuthStyle.AUTO:
            return AuthStyle.X_API_KEY
        return style

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
        }
        style = self._resolve_auth_style()
        if style in (AuthStyle.BEARER, AuthStyle.BOTH):
            headers["Authorization"] = f"Bearer {self.api_key}"
        if style in (AuthStyle.X_API_KEY, AuthStyle.BOTH):
            headers["x-api-key"] = self.api_key
        return headers

    def _build_models_url(self) -> str:
        return join_url(self.config.base_url, "/v1/models")

    def _build_completion_url(self) -> str:
        return join_url(self.config.base_url, "/v1/messages")

    def _build_completion_body(self, model: str, messages: list[dict[str, str]]) -> dict[str, object]:
        # Anthropic 消息格式转换
        system_prompt: str | None = None
        anthropic_messages: list[dict[str, object]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            else:
                anthropic_messages.append({"role": role, "content": content})

        body: dict[str, object] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": self.config.max_output_tokens,
        }
        if system_prompt:
            body["system"] = system_prompt
        return body

    def _apply_options_to_body(self, body: dict[str, object], options: CompletionOptions) -> None:
        """Map CompletionOptions to Anthropic-compatible request body keys.

        ================ ================
        CompletionOptions Anthropic body key
        ================ ================
        stop / until       stop_sequences
        temperature        temperature
        max_tokens /       max_tokens
        max_gen_toks
        do_sample          (validated only)
        ================ ================
        """
        stop_seqs = _resolve_stop_sequences(options.until, options.stop)
        if stop_seqs:
            body["stop_sequences"] = stop_seqs

        if options.temperature is not None:
            body["temperature"] = options.temperature

        max_toks = _resolve_max_tokens(options.max_gen_toks, options.max_tokens)
        if max_toks is not None:
            body["max_tokens"] = max_toks

        _validate_do_sample(options.do_sample)

    def _build_stream_body(self, model: str, messages: list[dict[str, str]]) -> dict[str, object]:
        body = self._build_completion_body(model, messages)
        body["stream"] = True
        return body

    def _extract_models(self, data: dict[str, object]) -> list[str]:
        models: list[str] = []
        items = data.get("data", [])
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    model_id = item.get("id")
                    if isinstance(model_id, str):
                        models.append(model_id)
        return models

    def _parse_response(self, data: dict[str, object], evidence: HTTPEvidence) -> None:
        # 提取模型
        model = data.get("model")
        if isinstance(model, str):
            evidence.response_model = model

        # 提取 ID
        rid = data.get("id")
        if isinstance(rid, str):
            evidence.response_id = rid

        # 提取 usage
        usage = data.get("usage")
        if isinstance(usage, dict):
            usage_dict: dict[str, Any] = usage
            it = usage_dict.get("input_tokens")
            if isinstance(it, int):
                evidence.input_tokens = it
            ot = usage_dict.get("output_tokens")
            if isinstance(ot, int):
                evidence.output_tokens = ot

        # 提取 stop_reason
        sr = data.get("stop_reason")
        if isinstance(sr, str):
            evidence.finish_reason = sr

        # 提取 response text
        content_list = data.get("content")
        if isinstance(content_list, list):
            text_parts = []
            for block in content_list:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
            evidence.response_text = "".join(text_parts)

    def _parse_stream_event(self, line: str) -> dict[str, object] | None:
        line = line.strip()
        if not line or not line.startswith("data:"):
            return None
        data_str = line[5:].strip()
        try:
            parsed = json.loads(data_str)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return None

    def _extract_stream_text(self, event: dict[str, object]) -> str | None:
        event_type = event.get("type")
        if event_type == "content_block_delta":
            delta = event.get("delta")
            if isinstance(delta, dict):
                text = delta.get("text")
                if isinstance(text, str):
                    return text
        return None

    def _parse_stream_finish(self, event: dict[str, object], evidence: HTTPEvidence) -> None:
        event_type = event.get("type")
        if event_type == "message_start":
            msg = event.get("message")
            if isinstance(msg, dict):
                model = msg.get("model")
                if isinstance(model, str):
                    evidence.response_model = model
                rid = msg.get("id")
                if isinstance(rid, str):
                    evidence.response_id = rid
        elif event_type == "message_delta":
            delta = event.get("delta")
            if isinstance(delta, dict):
                sr = delta.get("stop_reason")
                if isinstance(sr, str):
                    evidence.finish_reason = sr
            usage = event.get("usage")
            if isinstance(usage, dict):
                usage_dict: dict[str, Any] = usage
                ot = usage_dict.get("output_tokens")
                if isinstance(ot, int):
                    evidence.output_tokens = ot
