"""Task 22 / Task 23 — the fingerprint cost profile inside the hard budget.

The request ceiling is the run's only hard spend guarantee, so the fingerprint
stage must be counted *before* the first request: probes x repetitions for the
chosen profile, added on top of protocol + benchmark, and never silently
dropped.  Turning identity evidence off must leave the plan byte-for-byte the
pre-v0.6 plan.
"""

from __future__ import annotations

import pytest

from llmtrace.adapters.quick_suite import QUICK_SUITE_BENCHMARK_REQUESTS
from llmtrace.config import AuditConfig
from llmtrace.execution.models import UnifiedExecutionPlan
from llmtrace.execution.planner import build_unified_execution_plan
from llmtrace.execution.protocol_audit import protocol_probe_request_count
from llmtrace.fingerprint.models import FingerprintProfile, repetitions_for_profile
from llmtrace.fingerprint.suite import load_fingerprint_suite

TARGET_ID = "openai-test-target"
SET_SHA_A = "a" * 64
SET_SHA_B = "b" * 64


def _plan(config: AuditConfig, **kwargs: object) -> UnifiedExecutionPlan:
    return build_unified_execution_plan(config, target_id=TARGET_ID, **kwargs)


def _fingerprint_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "fingerprint_profile": FingerprintProfile.STANDARD,
        "fingerprint_set_id": "test-identity-set",
        "fingerprint_set_version": "0.1.0",
        "fingerprint_set_content_sha256": SET_SHA_A,
    }
    kwargs.update(overrides)
    return kwargs


class TestFingerprintBudget:
    def test_profile_reserves_probes_times_repetitions(self, config: AuditConfig) -> None:
        suite = load_fingerprint_suite()
        plan = _plan(config, fingerprint_probe_count=len(suite.probes), **_fingerprint_kwargs())

        expected = len(suite.probes) * repetitions_for_profile(FingerprintProfile.STANDARD)
        assert expected == 48
        assert plan.fingerprint_requests == expected
        assert plan.planned_requests == (
            protocol_probe_request_count(config) + QUICK_SUITE_BENCHMARK_REQUESTS + expected
        )
        # The hard ceiling must cover every request the run can actually spend.
        assert plan.maximum_requests == plan.planned_requests
        assert plan.fingerprint_profile == FingerprintProfile.STANDARD.value

    @pytest.mark.parametrize(
        ("profile", "expected_repetitions"),
        [
            (FingerprintProfile.QUICK, 4),
            (FingerprintProfile.STANDARD, 8),
            (FingerprintProfile.RESEARCH, 16),
        ],
    )
    def test_each_profile_uses_its_own_cost_tier(
        self,
        config: AuditConfig,
        profile: FingerprintProfile,
        expected_repetitions: int,
    ) -> None:
        suite = load_fingerprint_suite()
        plan = _plan(
            config,
            fingerprint_probe_count=len(suite.probes),
            **_fingerprint_kwargs(fingerprint_profile=profile),
        )

        assert plan.fingerprint_requests == len(suite.probes) * expected_repetitions
        assert plan.planned_requests == (
            protocol_probe_request_count(config) + QUICK_SUITE_BENCHMARK_REQUESTS + plan.fingerprint_requests
        )

    def test_without_profile_the_plan_keeps_legacy_budget(self, config: AuditConfig) -> None:
        plan = _plan(config)

        assert plan.fingerprint_requests == 0
        assert plan.planned_requests == protocol_probe_request_count(config) + QUICK_SUITE_BENCHMARK_REQUESTS
        assert plan.maximum_requests == plan.planned_requests
        assert plan.fingerprint_profile is None
        assert plan.fingerprint_set_id is None
        assert plan.fingerprint_set_version is None
        assert plan.fingerprint_set_content_sha256 is None

    def test_profile_without_probe_count_is_rejected(self, config: AuditConfig) -> None:
        # A profile with zero planned requests would be an un-auditable plan:
        # the provenance bundle and the count must be set together.
        with pytest.raises(ValueError, match="fingerprint context and fingerprint_requests"):
            _plan(config, fingerprint_probe_count=0, **_fingerprint_kwargs())

    def test_probe_count_without_profile_is_ignored(self, config: AuditConfig) -> None:
        # No profile means identity evidence is off; a stray probe count must
        # not reserve requests the run would never spend.
        plan = _plan(config, fingerprint_probe_count=6)

        assert plan.fingerprint_requests == 0
        assert plan.fingerprint_profile is None


class TestPlanIdentity:
    def test_plan_id_binds_the_fingerprint_reference_identity(self, config: AuditConfig) -> None:
        plan_a = _plan(config, fingerprint_probe_count=6, **_fingerprint_kwargs())
        plan_b = _plan(
            config,
            fingerprint_probe_count=6,
            **_fingerprint_kwargs(fingerprint_set_content_sha256=SET_SHA_B),
        )

        # A different reference set is a different measurement → different identity.
        assert plan_a.plan_id != plan_b.plan_id

    def test_plan_id_binds_the_cost_profile(self, config: AuditConfig) -> None:
        standard = _plan(config, fingerprint_probe_count=6, **_fingerprint_kwargs())
        quick = _plan(
            config,
            fingerprint_probe_count=6,
            **_fingerprint_kwargs(fingerprint_profile=FingerprintProfile.QUICK),
        )

        assert standard.plan_id != quick.plan_id

    def test_plan_id_is_deterministic(self, config: AuditConfig) -> None:
        first = _plan(config, fingerprint_probe_count=6, **_fingerprint_kwargs())
        second = _plan(config, fingerprint_probe_count=6, **_fingerprint_kwargs())

        assert first.plan_id == second.plan_id

    def test_fingerprint_plan_never_collides_with_a_capability_only_plan(self, config: AuditConfig) -> None:
        assert _plan(config).plan_id != _plan(config, fingerprint_probe_count=6, **_fingerprint_kwargs()).plan_id
