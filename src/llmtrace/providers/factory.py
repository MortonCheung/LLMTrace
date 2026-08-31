"""Provider factory — the single implementation, shared by CLI and services."""

from __future__ import annotations

from llmtrace.config import AuditConfig, Protocol
from llmtrace.execution.budget import RequestBudget
from llmtrace.execution.evidence import EvidenceRecorder
from llmtrace.providers.anthropic_compatible import AnthropicCompatibleProvider
from llmtrace.providers.base import BaseProvider
from llmtrace.providers.openai_compatible import OpenAICompatibleProvider


def create_provider(
    config: AuditConfig,
    api_key: str,
    *,
    evidence_recorder: EvidenceRecorder | None = None,
    request_budget: RequestBudget | None = None,
) -> BaseProvider:
    """Create a provider for *config*, wiring in optional execution hooks.

    The service layer (and the unified runner) must use this factory instead
    of keeping a second provider construction path in the CLI.
    """
    if config.protocol == Protocol.OPENAI:
        return OpenAICompatibleProvider(
            config,
            api_key,
            evidence_recorder=evidence_recorder,
            request_budget=request_budget,
        )
    if config.protocol == Protocol.ANTHROPIC:
        return AnthropicCompatibleProvider(
            config,
            api_key,
            evidence_recorder=evidence_recorder,
            request_budget=request_budget,
        )
    raise ValueError(f"不支持的协议: {config.protocol}")
