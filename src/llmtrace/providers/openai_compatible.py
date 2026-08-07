"""OpenAI-compatible Provider."""

from __future__ import annotations

import json
from typing import Any

from llmtrace.benchmarks.models import CompletionOptions
from llmtrace.config import AuditConfig
from llmtrace.models.evidence import HTTPEvidence
from llmtrace.providers.base import (
    BaseProvider,
    _resolve_max_tokens,
    _resolve_stop_sequences,
    _validate_do_sample,
)
from llmtrace.providers.url_utils import join_url


class OpenAICompatibleProvider(BaseProvider):
    """OpenAI-compatible 协议 Provider."""

    def __init__(self, config: AuditConfig, api_key: str) -> None:
        super().__init__(config, api_key)

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_models_url(self) -> str:
        return join_url(self.config.base_url, "/v1/models")

    def _build_completion_url(self) -> str:
        return join_url(self.config.base_url, "/v1/chat/completions")

    def _build_completion_body(self, model: str, messages: list[dict[str, str]]) -> dict[str, object]:
        return {
            "model": model,
            "messages": messages,
            "max_tokens": self.config.max_output_tokens,
        }

    def _apply_options_to_body(self, body: dict[str, object], options: CompletionOptions) -> None:
        """Map CompletionOptions to OpenAI-compatible request body keys.

        ================ ==============
        CompletionOptions OpenAI body key
        ================ ==============
        stop / until       stop
        temperature        temperature
        max_tokens /       max_tokens
        max_gen_toks
        do_sample          (validated only)
        ================ ==============
        """
        stop_seqs = _resolve_stop_sequences(options.until, options.stop)
        if stop_seqs:
            body["stop"] = stop_seqs

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
            # 使用 Any 类型来避免 mypy 严格检查
            usage_dict: dict[str, Any] = usage
            it = usage_dict.get("prompt_tokens")
            if isinstance(it, int):
                evidence.input_tokens = it
            ot = usage_dict.get("completion_tokens")
            if isinstance(ot, int):
                evidence.output_tokens = ot

        # 提取 choices
        choices = data.get("choices")
        if isinstance(choices, list) and len(choices) > 0:
            first = choices[0]
            if isinstance(first, dict):
                finish = first.get("finish_reason")
                if isinstance(finish, str):
                    evidence.finish_reason = finish
                msg = first.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        evidence.response_text = content

    def _parse_stream_event(self, line: str) -> dict[str, object] | None:
        line = line.strip()
        if not line or not line.startswith("data:"):
            return None
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            return None
        try:
            parsed = json.loads(data_str)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return None

    def _extract_stream_text(self, event: dict[str, object]) -> str | None:
        choices = event.get("choices")
        if isinstance(choices, list) and len(choices) > 0:
            first = choices[0]
            if isinstance(first, dict):
                delta = first.get("delta")
                if isinstance(delta, dict):
                    content = delta.get("content")
                    if isinstance(content, str):
                        return content
        return None

    def _parse_stream_finish(self, event: dict[str, object], evidence: HTTPEvidence) -> None:
        # 尝试从流事件中提取 finish_reason 和 usage
        model = event.get("model")
        if isinstance(model, str) and evidence.response_model is None:
            evidence.response_model = model

        rid = event.get("id")
        if isinstance(rid, str) and evidence.response_id is None:
            evidence.response_id = rid

        choices = event.get("choices")
        if isinstance(choices, list) and len(choices) > 0:
            first = choices[0]
            if isinstance(first, dict):
                finish = first.get("finish_reason")
                if isinstance(finish, str) and evidence.finish_reason is None:
                    evidence.finish_reason = finish

        usage = event.get("usage")
        if isinstance(usage, dict):
            usage_dict: dict[str, Any] = usage
            it = usage_dict.get("prompt_tokens")
            if isinstance(it, int) and evidence.input_tokens is None:
                evidence.input_tokens = it
            ot = usage_dict.get("completion_tokens")
            if isinstance(ot, int) and evidence.output_tokens is None:
                evidence.output_tokens = ot
