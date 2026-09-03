"""Adversarial tests for actual capability_profile.json bytes verification (v0.4-B).

These tests cover the final owner blocker: preflight must verify the actual
persisted capability_profile.json bytes against the trust chain, not just
compare SHA-256s across the provenance records.

Test Coverage:
    A. tampered actual profile bytes (manifest/sidecar/provenance intact)
    B. runner preflight rejects tampered profile with zero HTTP
    C. missing capability_profile.json artifact
    D. TOCTOU regression — verified profiles pinned in context

All tests use production builders via ``build_trusted_reference_root``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmtrace.config import AuditConfig, Protocol
from llmtrace.execution.artifacts import RunArtifactRepository
from llmtrace.execution.runner import PreflightError, UnifiedAuditRunner
from llmtrace.reference.validation import validate_reference_set_for_calibration
from llmtrace.scoring.calibration import ReferenceSetIntegrityFailureError
from tests.execution.conftest import TrustedFakeBackend

from .helpers import build_trusted_reference_root


def _artifact_repository(root: Path) -> RunArtifactRepository:
    return RunArtifactRepository(root)


@pytest.fixture
def api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "test-key")


def _make_config() -> AuditConfig:
    return AuditConfig(
        protocol=Protocol.OPENAI,
        base_url="http://test.example.com/v1",
        model="demo-model",
        api_key_env="TEST_KEY",
        repeat_count=1,
        max_output_tokens=64,
        check_streaming=False,
    )


class TestCapabilityProfileBytesVerification:
    def test_tampered_actual_profile_bytes_rejected(self, tmp_path: Path) -> None:
        """A: tamper actual capability_profile.json bytes, leave manifest/sidecar
        intact — validation must detect the actual bytes hash mismatch."""
        ref_root, set_path = build_trusted_reference_root(tmp_path)
        reference_set_json = json.loads(set_path.read_text(encoding="utf-8"))
        first_member = reference_set_json["members"][0]
        execution_id = first_member["execution_id"]

        # Tamper the actual capability_profile.json bytes
        profile_path = tmp_path / "runs" / execution_id / "capability_profile.json"
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        profile["provisional_raw_index"] = 0.9999  # corrupt actual bytes
        profile_path.write_text(json.dumps(profile, indent=2))

        # Validation must reject: actual bytes SHA != expected SHA in manifest
        with pytest.raises(
            ReferenceSetIntegrityFailureError,
            match="capability_profile.json could not be loaded or verified",
        ):
            validate_reference_set_for_calibration(
                set_path=set_path,
                artifact_repository=_artifact_repository(tmp_path),
            )

    def test_runner_preflight_rejects_tampered_profile_zero_http(
        self, tmp_path: Path, api_key_env: pytest.MonkeyPatch
    ) -> None:
        """B: runner preflight must reject tampered profile before any target HTTP."""
        ref_root, set_path = build_trusted_reference_root(tmp_path)
        reference_set_json = json.loads(set_path.read_text(encoding="utf-8"))
        first_member = reference_set_json["members"][0]
        execution_id = first_member["execution_id"]

        # Tamper actual bytes
        profile_path = tmp_path / "runs" / execution_id / "capability_profile.json"
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        profile["provisional_raw_index"] = 0.9999
        profile_path.write_text(json.dumps(profile, indent=2))

        # Create runner with reference_set
        config = _make_config()
        repository = _artifact_repository(tmp_path)
        runner = UnifiedAuditRunner(
            code_backend=TrustedFakeBackend(),
            config=config,
            api_key="sk-fake",
            target_id="test-target",
            repository=repository,
            reference_set_path=set_path,
        )

        # Must raise PreflightError (zero HTTP, zero provider instantiation)
        with pytest.raises(PreflightError, match="reference set rejected for formal calibration"):
            import asyncio

            asyncio.run(runner.run())

    def test_missing_capability_artifact_rejected(self, tmp_path: Path) -> None:
        """C: delete capability_profile.json, leave manifest intact — must reject."""
        ref_root, set_path = build_trusted_reference_root(tmp_path)
        reference_set_json = json.loads(set_path.read_text(encoding="utf-8"))
        first_member = reference_set_json["members"][0]
        execution_id = first_member["execution_id"]

        # Delete actual artifact
        profile_path = tmp_path / "runs" / execution_id / "capability_profile.json"
        profile_path.unlink()

        # Must reject with integrity failure (not ArtifactNotFoundError leaking)
        with pytest.raises(
            ReferenceSetIntegrityFailureError,
            match="capability_profile.json could not be loaded or verified",
        ):
            validate_reference_set_for_calibration(
                set_path=set_path,
                artifact_repository=_artifact_repository(tmp_path),
            )

    def test_verified_profiles_pinned_in_context(self, tmp_path: Path) -> None:
        """D: CalibrationContext.verified_profiles must contain all member profiles
        loaded during preflight, preventing TOCTOU between validation and calibration."""
        ref_root, set_path = build_trusted_reference_root(tmp_path)
        context = validate_reference_set_for_calibration(
            set_path=set_path,
            artifact_repository=_artifact_repository(tmp_path),
        )

        # All 5 members must have verified profiles pinned in context
        assert len(context.verified_profiles) == 5
        assert all(
            snap_id in context.verified_profiles
            for member in context.reference_set.members
            for snap_id in [member.snapshot_id]
        )

        # Each profile must be the actual CapabilityProfile, not None
        for _snap_id, profile in context.verified_profiles.items():
            assert profile is not None
            assert hasattr(profile, "provisional_raw_index")
            assert hasattr(profile, "dimensions")

        # TOCTOU regression: tamper disk after validation
        reference_set_json = json.loads(set_path.read_text(encoding="utf-8"))
        first_member = reference_set_json["members"][0]
        execution_id = first_member["execution_id"]
        profile_path = tmp_path / "runs" / execution_id / "capability_profile.json"
        profile_json = json.loads(profile_path.read_text(encoding="utf-8"))
        original_raw_index = profile_json["provisional_raw_index"]
        profile_json["provisional_raw_index"] = 0.8888
        profile_path.write_text(json.dumps(profile_json, indent=2))

        # Context profiles must still have the pre-validation values
        first_snap_id = context.reference_set.members[0].snapshot_id
        verified_profile = context.verified_profiles[first_snap_id]
        assert verified_profile.provisional_raw_index != 0.8888
        assert verified_profile.provisional_raw_index == original_raw_index
