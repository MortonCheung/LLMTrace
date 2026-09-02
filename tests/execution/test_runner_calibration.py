"""Tests for Runner calibration integration — v0.4-B."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from llmtrace.adapters.quick_suite import (
    QUICK_SUITE_SUITE_ID,
    QUICK_SUITE_SUITE_VERSION,
    get_quick_suite_content_sha256,
)
from llmtrace.execution.artifacts import RunArtifactRepository
from llmtrace.execution.runner import PreflightError, UnifiedAuditRunner
from llmtrace.reference.reference_set import ReferenceSet, ReferenceSetMember
from llmtrace.scoring.models import CapabilityProfile
from llmtrace.scoring.policy import CapabilityScoringPolicy
from tests.execution.conftest import TrustedFakeBackend
from tests.reference.helpers import (
    expected_generation_config_sha256,
    make_capability_profile,
    make_manifest,
)

_POLICY = CapabilityScoringPolicy.create_v1()


def _make_profile(score: float = 0.5) -> CapabilityProfile:
    return make_capability_profile(score)


def _make_config():
    from llmtrace.config import AuditConfig, Protocol

    return AuditConfig(
        protocol=Protocol.OPENAI,
        base_url="http://test.example.com/v1",
        model="demo-model",
        api_key_env="TEST_KEY",
        repeat_count=1,
        max_output_tokens=64,
        check_streaming=False,
    )


_UUID_SEED = "aaaaaaaa-bbbb-cccc-dddd-eeee"


def _build_ref_set(
    member_count: int = 5,
    artifact_root: Path | None = None,
) -> tuple[ReferenceSet, list[str]]:
    """Build a ReferenceSet with member execution directories."""
    members: list[ReferenceSetMember] = []
    exec_ids: list[str] = []
    for i in range(member_count):
        snap_id = f"snap-{i}"
        exec_id = f"{_UUID_SEED}{i:04d}"
        members.append(
            ReferenceSetMember(
                snapshot_id=snap_id,
                snapshot_sha256="a" * 64,
                model_id=f"model-{i}",
                provider_id=f"provider-{i}",
                execution_id=exec_id,
                capability_profile_sha256="c" * 64,
            )
        )
        exec_ids.append(exec_id)
        if artifact_root is not None:
            repo = RunArtifactRepository(artifact_root)
            manifest = make_manifest(execution_id=exec_id)
            profile = _make_profile(score=0.3 + i * 0.1)
            repo.commit(manifest, {"capability_profile.json": profile.model_dump_json()})

    ref_set = ReferenceSet.model_construct(
        reference_set_id="test-set",
        reference_set_version="1",
        created_at=datetime.now(UTC),
        suite_id=QUICK_SUITE_SUITE_ID,
        suite_version=QUICK_SUITE_SUITE_VERSION,
        suite_content_sha256=get_quick_suite_content_sha256(),
        adapter_id="llmtrace-quick-v1",
        adapter_version="0.1.0",
        scoring_policy_id=_POLICY.policy_id,
        scoring_policy_version=_POLICY.policy_version,
        generation_config_sha256=expected_generation_config_sha256(),
        qualification_policy_id="llmtrace_reference_qualification_v1",
        qualification_policy_version="0.1.0",
        members=tuple(members),
        content_sha256="",
    )
    ref_set = ref_set.model_copy(update={"content_sha256": ref_set.compute_content_sha256()})
    return ref_set, exec_ids


class TestRunnerPreflightReferenceSet:
    def test_invalid_reference_set_file(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "bad_ref_set.json"
        bad_path.write_text("not valid json")
        repo = RunArtifactRepository(tmp_path)
        runner = UnifiedAuditRunner(
            _make_config(),
            api_key="test-key",
            target_id="test-target",
            repository=repo,
            code_backend=TrustedFakeBackend(),
            reference_set_path=bad_path,
        )
        with pytest.raises(PreflightError, match="reference set"):
            runner._preflight()

    def test_reference_set_content_hash_mismatch(self, tmp_path: Path) -> None:
        ref_set, _ = _build_ref_set(member_count=5, artifact_root=tmp_path)
        bad_path = tmp_path / "bad_ref_set.json"
        bad_set = ref_set.model_copy(update={"content_sha256": "0" * 64})
        bad_path.write_text(bad_set.model_dump_json())

        repo = RunArtifactRepository(tmp_path)
        runner = UnifiedAuditRunner(
            _make_config(),
            api_key="test-key",
            target_id="test-target",
            repository=repo,
            code_backend=TrustedFakeBackend(),
            reference_set_path=bad_path,
        )
        with pytest.raises(PreflightError, match="integrity failure"):
            runner._preflight()

    def test_valid_reference_set_sets_policy(self, tmp_path: Path) -> None:
        ref_set, _ = _build_ref_set(member_count=5, artifact_root=tmp_path)
        ref_set_path = tmp_path / "ref_set.json"
        ref_set_path.write_text(ref_set.model_dump_json())

        repo = RunArtifactRepository(tmp_path)
        runner = UnifiedAuditRunner(
            _make_config(),
            api_key="test-key",
            target_id="test-target",
            repository=repo,
            code_backend=TrustedFakeBackend(),
            reference_set_path=ref_set_path,
        )
        runner._preflight()
        assert runner._reference_set is not None
        assert runner._reference_set.reference_set_id == "test-set"
        assert runner._calibration_policy is not None
