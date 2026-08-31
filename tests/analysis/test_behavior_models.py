"""Tests for Behavior Drift domain models (behavior_models.py)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from llmtrace.analysis.behavior_models import (
    BehaviorDriftPolicy,
    BehaviorItemKey,
    BehaviorItemObservation,
    BehaviorRunSnapshot,
    canonicalize_output,
    generation_config_sha256,
    output_text_sha256,
)
from llmtrace.benchmarks.models import ItemStatus

from .conftest import (
    _sha,
    make_profile,
    make_snapshot,
)

_VALID_SHA = _sha("test")


# ===========================================================================
# BehaviorItemKey
# ===========================================================================


class TestBehaviorItemKey:
    def test_key_is_frozen(self) -> None:
        key = BehaviorItemKey(task_id="t", source_sample_id="s", input_sha256=_VALID_SHA)
        with pytest.raises((TypeError, ValueError)):
            key.task_id = "other"  # type: ignore[misc]

    def test_key_hashable_and_usable_as_dict_key(self) -> None:
        key = BehaviorItemKey(task_id="t", source_sample_id="s", input_sha256=_VALID_SHA)
        d = {key: "value"}
        assert d[BehaviorItemKey(task_id="t", source_sample_id="s", input_sha256=_VALID_SHA)] == "value"

    def test_sort_key_is_deterministic(self) -> None:
        key = BehaviorItemKey(task_id="t", source_sample_id="s", input_sha256=_VALID_SHA)
        assert key.sort_key == ("t", "s", _VALID_SHA)

    @pytest.mark.parametrize("empty", ["", "   "])
    def test_empty_task_id_rejected(self, empty: str) -> None:
        with pytest.raises(ValueError):
            BehaviorItemKey(task_id=empty, source_sample_id="s", input_sha256=_VALID_SHA)

    @pytest.mark.parametrize("empty", ["", "   "])
    def test_empty_source_sample_id_rejected(self, empty: str) -> None:
        with pytest.raises(ValueError):
            BehaviorItemKey(task_id="t", source_sample_id=empty, input_sha256=_VALID_SHA)

    @pytest.mark.parametrize(
        "bad",
        ["abc123", "", "z" * 64, "a" * 63, "a" * 65, _VALID_SHA.upper()],
    )
    def test_invalid_input_sha256_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError):
            BehaviorItemKey(task_id="t", source_sample_id="s", input_sha256=bad)

    def test_uppercase_sha256_rejected(self) -> None:
        with pytest.raises(ValueError):
            BehaviorItemKey(task_id="t", source_sample_id="s", input_sha256=_VALID_SHA.upper())


# ===========================================================================
# Output canonicalization
# ===========================================================================


class TestOutputCanonicalization:
    def test_crlf_and_lf_same(self) -> None:
        assert output_text_sha256("The answer is 42.\r\n") == output_text_sha256("The answer is 42.\n")

    def test_trailing_newline_same(self) -> None:
        assert output_text_sha256("The answer is 42.\n") == output_text_sha256("The answer is 42.")

    def test_leading_and_trailing_whitespace_same(self) -> None:
        assert output_text_sha256("  The answer is 42.  ") == output_text_sha256("The answer is 42.")

    def test_internal_change_differs(self) -> None:
        assert output_text_sha256("The answer is 42.") != output_text_sha256("42")

    def test_empty_output_has_defined_hash(self) -> None:
        assert output_text_sha256("") == output_text_sha256("   \n")

    def test_canonicalize_does_not_lowercase(self) -> None:
        assert canonicalize_output("ANSWER") != canonicalize_output("answer")


# ===========================================================================
# Generation config hash
# ===========================================================================


class TestGenerationConfigSha256:
    def test_dict_key_order_independent(self) -> None:
        a = generation_config_sha256({"temperature": 0.0, "max_tokens": 512})
        b = generation_config_sha256({"max_tokens": 512, "temperature": 0.0})
        assert a == b

    def test_different_values_differ(self) -> None:
        a = generation_config_sha256({"temperature": 0.0})
        b = generation_config_sha256({"temperature": 0.5})
        assert a != b

    def test_deterministic(self) -> None:
        assert generation_config_sha256({"temperature": 0.0}) == generation_config_sha256({"temperature": 0.0})

    def test_rejects_non_mapping(self) -> None:
        with pytest.raises(ValueError):
            generation_config_sha256(42)  # type: ignore[arg-type]

    def test_accepts_pydantic_model(self) -> None:
        from llmtrace.benchmarks.models import CompletionOptions

        digest = generation_config_sha256(CompletionOptions(temperature=0.0, max_tokens=512))
        assert len(digest) == 64
        assert isinstance(digest, str)


# ===========================================================================
# BehaviorItemObservation
# ===========================================================================


class TestBehaviorItemObservation:
    def test_observation_frozen(self) -> None:
        obs = BehaviorItemObservation(
            key=BehaviorItemKey(task_id="t", source_sample_id="s", input_sha256=_VALID_SHA),
            status=ItemStatus.GRADED,
            raw_score=1.0,
            normalized_score=1.0,
            output_text_sha256=output_text_sha256("42"),
            output_length=2,
        )
        with pytest.raises((TypeError, ValueError)):
            obs.normalized_score = 0.5  # type: ignore[misc]

    def test_output_hash_must_be_valid_sha(self) -> None:
        with pytest.raises(ValueError):
            BehaviorItemObservation(
                key=BehaviorItemKey(task_id="t", source_sample_id="s", input_sha256=_VALID_SHA),
                status=ItemStatus.GRADED,
                raw_score=1.0,
                normalized_score=1.0,
                output_text_sha256="not-a-hash",
                output_length=2,
            )


# ===========================================================================
# BehaviorRunSnapshot
# ===========================================================================


class TestBehaviorRunSnapshot:
    def test_snapshot_frozen(self) -> None:
        snap = make_snapshot()
        with pytest.raises((TypeError, ValueError)):
            snap.run_id = "other"  # type: ignore[misc]

    def test_naive_created_at_rejected(self) -> None:

        with pytest.raises(ValueError, match="timezone-aware"):
            BehaviorRunSnapshot(
                run_id="r",
                target_id="t",
                candidate_model_id="m",
                created_at=datetime(2026, 8, 31),
                suite_id="s",
                suite_version="1",
                adapter_id="a",
                adapter_version="1",
                scoring_policy_id="p",
                scoring_policy_version="1",
                generation_config_sha256=_VALID_SHA,
                capability_profile=make_profile(),
            )

    def test_utc_normalization(self) -> None:

        plus_eight = timezone(timedelta(hours=8))
        snap = BehaviorRunSnapshot(
            run_id="r",
            target_id="t",
            candidate_model_id="m",
            created_at=datetime(2026, 8, 31, 8, 0, tzinfo=plus_eight),
            suite_id="s",
            suite_version="1",
            adapter_id="a",
            adapter_version="1",
            scoring_policy_id="p",
            scoring_policy_version="1",
            generation_config_sha256=_VALID_SHA,
            capability_profile=make_profile(),
        )
        assert snap.created_at.tzinfo is UTC
        assert snap.created_at == datetime(2026, 8, 31, 0, 0, tzinfo=UTC)

    def test_invalid_generation_config_sha256_rejected(self) -> None:

        with pytest.raises(ValueError):
            BehaviorRunSnapshot(
                run_id="r",
                target_id="t",
                candidate_model_id="m",
                created_at=datetime(2026, 8, 31, tzinfo=UTC),
                suite_id="s",
                suite_version="1",
                adapter_id="a",
                adapter_version="1",
                scoring_policy_id="p",
                scoring_policy_version="1",
                generation_config_sha256="abc123",
                capability_profile=make_profile(),
            )


# ===========================================================================
# BehaviorDriftPolicy
# ===========================================================================


class TestBehaviorDriftPolicy:
    def test_create_v1_defaults(self) -> None:
        policy = BehaviorDriftPolicy.create_v1()
        assert policy.policy_id == "llmtrace_behavior_drift_v1"
        assert policy.policy_version == "0.1.0"
        assert 0.0 <= policy.minimum_graded_overlap_ratio <= 1.0
        assert 0.0 < policy.material_dimension_delta <= 1.0
        assert 0.0 <= policy.material_outcome_change_ratio <= 1.0

    def test_policy_frozen(self) -> None:
        policy = BehaviorDriftPolicy.create_v1()
        with pytest.raises((TypeError, ValueError)):
            policy.material_dimension_delta = 0.5  # type: ignore[misc]

    def test_invalid_thresholds_rejected(self) -> None:
        with pytest.raises(ValueError):
            BehaviorDriftPolicy(
                policy_id="p",
                policy_version="1",
                minimum_graded_overlap_ratio=-0.1,
                material_dimension_delta=0.2,
                material_outcome_change_ratio=0.3,
            )
        with pytest.raises(ValueError):
            BehaviorDriftPolicy(
                policy_id="p",
                policy_version="1",
                minimum_graded_overlap_ratio=0.5,
                material_dimension_delta=0.0,  # must be > 0
                material_outcome_change_ratio=0.3,
            )
