"""Unified execution planner — the full request/token plan before any call."""

from __future__ import annotations

import hashlib
import json

from llmtrace.adapters.quick_suite import (
    QUICK_SUITE_BENCHMARK_REQUESTS,
    QUICK_SUITE_SUITE_ID,
    QUICK_SUITE_SUITE_VERSION,
    get_quick_suite_content_sha256,
    get_quick_suite_generation_config,
    get_quick_suite_generation_config_sha256,
)
from llmtrace.config import AuditConfig
from llmtrace.execution.models import UnifiedExecutionPlan
from llmtrace.execution.protocol_audit import (
    protocol_output_token_ceiling,
    protocol_probe_request_count,
)
from llmtrace.scoring.policy import CapabilityScoringPolicy
from llmtrace.security.redaction import redact_url


def derive_target_id(protocol: str, base_url: str) -> str:
    """Derive a stable, non-secret target id from protocol + canonical endpoint.

    Never mixes the API key or any credential into the digest — the URL is
    redacted (userinfo credentials and secret query values stripped) before
    hashing.  The result is filesystem-safe: ``<protocol>-<12 hex>``.
    """
    canonical = redact_url(base_url).rstrip("/").lower()
    digest = hashlib.sha256(f"{protocol}|{canonical}".encode()).hexdigest()[:12]
    return f"{protocol}-{digest}"


def sanitize_target_id(target_id: str) -> str:
    """Strip and validate a user-supplied target id (filesystem-safe)."""
    stripped = target_id.strip()
    if not stripped:
        raise ValueError("target_id must not be empty")
    forbidden = set("/\\\0")
    if any(ch in forbidden for ch in stripped) or stripped in (".", ".."):
        raise ValueError(f"target_id must be path-safe, got {target_id!r}")
    return stripped


def build_unified_execution_plan(
    config: AuditConfig,
    *,
    target_id: str,
    policy: CapabilityScoringPolicy | None = None,
    reference_set_id: str | None = None,
    reference_set_version: str | None = None,
    reference_set_content_sha256: str | None = None,
    calibration_policy_id: str | None = None,
    calibration_policy_version: str | None = None,
) -> UnifiedExecutionPlan:
    """Build the complete execution plan before any API request is sent.

    When reference_set / calibration parameters are provided the plan_id
    binds them, so a different ReferenceSet or calibration policy produces
    a different artifact identity.
    """
    resolved_policy = policy if policy is not None else CapabilityScoringPolicy.create_v1()

    protocol_requests = protocol_probe_request_count(config)
    benchmark_requests = QUICK_SUITE_BENCHMARK_REQUESTS
    planned_requests = protocol_requests + benchmark_requests

    token_ceiling = protocol_output_token_ceiling(config) + benchmark_requests * 512

    generation_config = get_quick_suite_generation_config()
    # Authoritative digest — the same value the reference qualification and
    # ReferenceSet compatibility gates compare against.
    generation_sha = get_quick_suite_generation_config_sha256()

    suite_content_sha256 = get_quick_suite_content_sha256()

    plan_id_input: dict[str, object] = {
        "protocol": str(config.protocol.value),
        "base_url": redact_url(config.base_url),
        "model": config.model,
        "repeat_count": config.repeat_count,
        "check_streaming": config.check_streaming,
        "max_output_tokens": config.max_output_tokens,
        "suite_id": QUICK_SUITE_SUITE_ID,
        "suite_version": QUICK_SUITE_SUITE_VERSION,
        "suite_content_sha256": suite_content_sha256,
        "generation_config": generation_config,
    }
    if reference_set_content_sha256 is not None:
        plan_id_input["reference_set_content_sha256"] = reference_set_content_sha256
    if calibration_policy_id is not None:
        plan_id_input["calibration_policy_id"] = calibration_policy_id
    if calibration_policy_version is not None:
        plan_id_input["calibration_policy_version"] = calibration_policy_version

    plan = UnifiedExecutionPlan(
        plan_id=hashlib.sha256(
            json.dumps(
                plan_id_input,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest(),
        target_id=target_id,
        candidate_model_id=config.model,
        protocol_probe_requests=protocol_requests,
        benchmark_requests=benchmark_requests,
        planned_requests=planned_requests,
        maximum_requests=planned_requests,
        maximum_output_token_ceiling=token_ceiling,
        estimated_cost=None,
        suite_id=QUICK_SUITE_SUITE_ID,
        suite_version=QUICK_SUITE_SUITE_VERSION,
        suite_content_sha256=suite_content_sha256,
        scoring_policy_id=resolved_policy.policy_id,
        scoring_policy_version=resolved_policy.policy_version,
        generation_config_sha256=generation_sha,
        requires_secure_code_sandbox=True,
        calibration_policy_id=calibration_policy_id,
        calibration_policy_version=calibration_policy_version,
        reference_set_id=reference_set_id,
        reference_set_version=reference_set_version,
        reference_set_content_sha256=reference_set_content_sha256,
    )
    return plan
