"""Tests for the Behavior Drift engine (behavior_drift.py)."""

from __future__ import annotations

import pytest

from llmtrace.analysis.behavior_drift import (
    BehaviorAdapterMismatchError,
    BehaviorCoverageMismatchError,
    BehaviorDriftEngine,
    BehaviorDriftLevel,
    BehaviorDriftResult,
    BehaviorItemSetMismatchError,
    BehaviorScoringPolicyMismatchError,
    BehaviorSourceMismatchError,
    BehaviorSuiteMismatchError,
    BehaviorSuiteVersionMismatchError,
    GenerationConfigMismatchError,
)
from llmtrace.analysis.behavior_models import BehaviorDriftPolicy
from llmtrace.benchmarks.models import ItemStatus
from llmtrace.scoring.models import CapabilityDimension, DimensionScoreStatus

from .conftest import (
    _sha,
    make_profile,
    make_snapshot,
)

_ENGINE = BehaviorDriftEngine()
_POLICY = BehaviorDriftPolicy.create_v1()

_4_ITEMS = [
    {"task_id": "gsm8k_quick_v1", "source_sample_id": f"s{i}", "status": ItemStatus.GRADED, "score": 1.0}
    for i in range(4)
]


def _compare(baseline, current, policy=_POLICY) -> BehaviorDriftResult:
    return _ENGINE.compare(baseline, current, policy)


# ===========================================================================
# Compatibility gate — fail closed
# ===========================================================================


class TestCompatibilityGate:
    def test_suite_id_mismatch(self) -> None:
        a = make_snapshot(items=_4_ITEMS, suite_id="llmtrace_quick_v1")
        b = make_snapshot(items=_4_ITEMS, suite_id="llmtrace_quick_v2")
        with pytest.raises(BehaviorSuiteMismatchError):
            _compare(a, b)

    def test_suite_version_mismatch(self) -> None:
        a = make_snapshot(items=_4_ITEMS, suite_version="0.1.0")
        b = make_snapshot(items=_4_ITEMS, suite_version="0.2.0")
        with pytest.raises(BehaviorSuiteVersionMismatchError):
            _compare(a, b)

    def test_source_revision_mismatch(self) -> None:
        a = make_snapshot(items=_4_ITEMS, source_revision="gsm8k-main-2023")
        b = make_snapshot(items=_4_ITEMS, source_revision="gsm8k-main-2024")
        with pytest.raises(BehaviorSourceMismatchError):
            _compare(a, b)

    def test_adapter_mismatch(self) -> None:
        a = make_snapshot(items=_4_ITEMS, adapter_id="llmtrace-quick-v1")
        b = make_snapshot(items=_4_ITEMS, adapter_id="llmtrace-quick-v2")
        with pytest.raises(BehaviorAdapterMismatchError):
            _compare(a, b)

    def test_scoring_policy_id_mismatch(self) -> None:
        a = make_snapshot(items=_4_ITEMS, profile=make_profile())
        b = make_snapshot(
            items=_4_ITEMS,
            profile=make_profile(),
        )
        b = b.model_copy(update={"scoring_policy_id": "another-policy"})
        with pytest.raises(BehaviorScoringPolicyMismatchError):
            _compare(a, b)

    def test_scoring_policy_version_mismatch(self) -> None:
        a = make_snapshot(items=_4_ITEMS)
        b = make_snapshot(items=_4_ITEMS).model_copy(update={"scoring_policy_version": "0.2.0"})
        with pytest.raises(BehaviorScoringPolicyMismatchError):
            _compare(a, b)

    def test_generation_config_mismatch(self) -> None:
        a = make_snapshot(items=_4_ITEMS, generation_config={"temperature": 0.0})
        b = make_snapshot(items=_4_ITEMS, generation_config={"temperature": 0.5})
        with pytest.raises(GenerationConfigMismatchError):
            _compare(a, b)

    def test_item_set_mismatch(self) -> None:
        a = make_snapshot(items=_4_ITEMS)
        b = make_snapshot(items=[_4_ITEMS[0], _4_ITEMS[1], _4_ITEMS[2]])  # 3 vs 4
        with pytest.raises(BehaviorItemSetMismatchError):
            _compare(a, b)

    def test_same_sample_but_changed_input_sha256(self) -> None:
        items_a = [{"task_id": "gsm8k_quick_v1", "source_sample_id": "s1", "status": ItemStatus.GRADED, "score": 1.0}]
        items_b = [
            {
                "task_id": "gsm8k_quick_v1",
                "source_sample_id": "s1",
                "input_sha256": _sha("modified-prompt"),
                "status": ItemStatus.GRADED,
                "score": 1.0,
            }
        ]
        a = make_snapshot(items=items_a)
        b = make_snapshot(items=items_b)
        with pytest.raises(BehaviorItemSetMismatchError):
            _compare(a, b)

    def test_incompatible_dimension_coverage(self) -> None:
        a = make_snapshot(items=_4_ITEMS, profile=make_profile())
        b = make_snapshot(
            items=_4_ITEMS,
            profile=make_profile(
                statuses={CapabilityDimension.MATH_SCIENCE: DimensionScoreStatus.UNAVAILABLE},
                coverage_weight=0.60,
            ),
        )
        with pytest.raises(BehaviorCoverageMismatchError):
            _compare(a, b)


# ===========================================================================
# Alignment
# ===========================================================================


class TestAlignment:
    def test_different_item_order_aligns_correctly(self) -> None:
        items_a = [
            {"task_id": "t", "source_sample_id": "a", "status": ItemStatus.GRADED, "score": 1.0},
            {"task_id": "t", "source_sample_id": "b", "status": ItemStatus.GRADED, "score": 1.0},
            {"task_id": "t", "source_sample_id": "c", "status": ItemStatus.GRADED, "score": 1.0},
        ]
        items_b = [
            {"task_id": "t", "source_sample_id": "c", "status": ItemStatus.GRADED, "score": 0.0},
            {"task_id": "t", "source_sample_id": "a", "status": ItemStatus.GRADED, "score": 1.0},
            {"task_id": "t", "source_sample_id": "b", "status": ItemStatus.GRADED, "score": 1.0},
        ]
        a = make_snapshot(items=items_a)
        b = make_snapshot(items=items_b)
        result = _compare(a, b)
        by_sample = {item.key.source_sample_id: item for item in result.item_diffs}
        assert by_sample["c"].score_delta == pytest.approx(-1.0)
        assert by_sample["a"].score_delta == pytest.approx(0.0)
        assert by_sample["b"].score_delta == pytest.approx(0.0)


# ===========================================================================
# Drift classification
# ===========================================================================


class TestDriftClassification:
    def test_identical_runs_no_significant_drift(self) -> None:
        a = make_snapshot(items=_4_ITEMS)
        b = make_snapshot(items=_4_ITEMS)
        result = _compare(a, b)
        assert result.drift_level == BehaviorDriftLevel.NO_SIGNIFICANT_DRIFT
        assert result.outcome_changed_count == 0
        assert result.output_changed_count == 0
        assert result.status_changed_count == 0

    def test_output_change_but_same_score_is_observed_not_material(self) -> None:
        base = [
            {
                "task_id": "gsm8k_quick_v1",
                "source_sample_id": f"s{i}",
                "status": ItemStatus.GRADED,
                "score": 1.0,
                "response_text": "The answer is 42.",
            }
            for i in range(4)
        ]
        current = [
            {
                "task_id": "gsm8k_quick_v1",
                "source_sample_id": f"s{i}",
                "status": ItemStatus.GRADED,
                "score": 1.0,
                "response_text": "Therefore, after calculation, the answer is 42.",
            }
            for i in range(4)
        ]
        result = _compare(make_snapshot(items=base), make_snapshot(items=current))
        assert result.output_changed_count == 4
        assert result.outcome_changed_count == 0
        assert result.drift_level == BehaviorDriftLevel.OBSERVED_DRIFT

    def test_correct_to_wrong_is_observed(self) -> None:
        base = _4_ITEMS
        current = [
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s0", "status": ItemStatus.GRADED, "score": 0.0},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s1", "status": ItemStatus.GRADED, "score": 1.0},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s2", "status": ItemStatus.GRADED, "score": 1.0},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s3", "status": ItemStatus.GRADED, "score": 1.0},
        ]
        result = _compare(make_snapshot(items=base), make_snapshot(items=current))
        assert result.outcome_changed_count == 1
        by_sample = {i.key.source_sample_id: i for i in result.item_diffs}
        assert by_sample["s0"].score_delta == pytest.approx(-1.0)
        assert by_sample["s0"].outcome_changed is True

    def test_wrong_to_correct_is_observed(self) -> None:
        base = [
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s0", "status": ItemStatus.GRADED, "score": 0.0},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s1", "status": ItemStatus.GRADED, "score": 1.0},
        ]
        current = [
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s0", "status": ItemStatus.GRADED, "score": 1.0},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s1", "status": ItemStatus.GRADED, "score": 1.0},
        ]
        result = _compare(make_snapshot(items=base), make_snapshot(items=current))
        by_sample = {i.key.source_sample_id: i for i in result.item_diffs}
        assert by_sample["s0"].score_delta == pytest.approx(1.0)

    def test_graded_to_failure_is_status_change(self) -> None:
        base = _4_ITEMS
        current = [
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s0", "status": ItemStatus.FAILURE},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s1", "status": ItemStatus.GRADED, "score": 1.0},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s2", "status": ItemStatus.GRADED, "score": 1.0},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s3", "status": ItemStatus.GRADED, "score": 1.0},
        ]
        result = _compare(make_snapshot(items=base), make_snapshot(items=current))
        assert result.status_changed_count == 1
        by_sample = {i.key.source_sample_id: i for i in result.item_diffs}
        assert by_sample["s0"].status_changed is True
        assert by_sample["s0"].current_status == ItemStatus.FAILURE

    def test_failure_to_graded_is_status_change(self) -> None:
        base = [
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s0", "status": ItemStatus.FAILURE},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s1", "status": ItemStatus.GRADED, "score": 1.0},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s2", "status": ItemStatus.GRADED, "score": 1.0},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s3", "status": ItemStatus.GRADED, "score": 1.0},
        ]
        current = _4_ITEMS
        result = _compare(make_snapshot(items=base), make_snapshot(items=current))
        assert result.status_changed_count == 1
        by_sample = {i.key.source_sample_id: i for i in result.item_diffs}
        assert by_sample["s0"].status_changed is True
        assert by_sample["s0"].current_status == ItemStatus.GRADED

    def test_ungradable_to_graded_is_status_change(self) -> None:
        base = [
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s0", "status": ItemStatus.UNGRADABLE},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s1", "status": ItemStatus.GRADED, "score": 1.0},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s2", "status": ItemStatus.GRADED, "score": 1.0},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s3", "status": ItemStatus.GRADED, "score": 1.0},
        ]
        current = _4_ITEMS
        result = _compare(make_snapshot(items=base), make_snapshot(items=current))
        assert result.status_changed_count == 1

    def test_insufficient_graded_overlap_is_inconclusive(self) -> None:
        base = _4_ITEMS
        current = [
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s0", "status": ItemStatus.FAILURE},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s1", "status": ItemStatus.FAILURE},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s2", "status": ItemStatus.FAILURE},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s3", "status": ItemStatus.GRADED, "score": 1.0},
        ]
        result = _compare(make_snapshot(items=base), make_snapshot(items=current))
        assert result.graded_overlap_count == 1
        assert result.graded_overlap_ratio == pytest.approx(0.25)
        assert result.drift_level == BehaviorDriftLevel.INCONCLUSIVE

    def test_material_dimension_delta(self) -> None:
        a = make_snapshot(items=_4_ITEMS, profile=make_profile())
        b = make_snapshot(
            items=_4_ITEMS,
            profile=make_profile({CapabilityDimension.MATH_SCIENCE: 0.7}),
        )
        result = _compare(a, b)
        by_dim = {d.dimension: d for d in result.dimension_diffs}
        assert by_dim[CapabilityDimension.MATH_SCIENCE].delta == pytest.approx(-0.3)
        assert by_dim[CapabilityDimension.MATH_SCIENCE].absolute_delta == pytest.approx(0.3)
        assert result.drift_level == BehaviorDriftLevel.MATERIAL_DRIFT

    def test_material_outcome_change_ratio(self) -> None:
        base = _4_ITEMS
        current = [
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s0", "status": ItemStatus.GRADED, "score": 0.0},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s1", "status": ItemStatus.GRADED, "score": 0.0},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s2", "status": ItemStatus.GRADED, "score": 1.0},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s3", "status": ItemStatus.GRADED, "score": 1.0},
        ]
        result = _compare(make_snapshot(items=base), make_snapshot(items=current))
        assert result.outcome_changed_count == 2
        assert result.outcome_changed_ratio == pytest.approx(0.5)
        assert result.drift_level == BehaviorDriftLevel.MATERIAL_DRIFT

    def test_operational_change_is_observed(self) -> None:
        base = [
            {
                "task_id": "gsm8k_quick_v1",
                "source_sample_id": "s0",
                "status": ItemStatus.GRADED,
                "score": 1.0,
                "response_model": "gpt-x",
                "finish_reason": "stop",
            }
        ]
        current = [
            {
                "task_id": "gsm8k_quick_v1",
                "source_sample_id": "s0",
                "status": ItemStatus.GRADED,
                "score": 1.0,
                "response_model": "gpt-4o",
                "finish_reason": "length",
            }
        ]
        result = _compare(make_snapshot(items=base), make_snapshot(items=current))
        assert result.response_model_change_count == 1
        assert result.finish_reason_change_count == 1
        item = result.item_diffs[0]
        assert item.operational_changed is True
        assert item.outcome_changed is False
        assert result.drift_level == BehaviorDriftLevel.OBSERVED_DRIFT


# ===========================================================================
# Failure semantics — no fake downgrade
# ===========================================================================


class TestNoFakeDowngrade:
    def test_provider_failure_not_converted_to_capability_delta(self) -> None:
        """A FAILURE item has score 0.0, but that must not read as a capability drop.

        Outcome drift only counts items graded in BOTH runs; a FAILURE item is
        excluded from the graded overlap, so its forced 0.0 never enters an
        outcome delta.  It shows up as a status change instead.
        """
        base = _4_ITEMS
        current = [
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s0", "status": ItemStatus.FAILURE},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s1", "status": ItemStatus.GRADED, "score": 1.0},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s2", "status": ItemStatus.GRADED, "score": 1.0},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s3", "status": ItemStatus.GRADED, "score": 1.0},
        ]
        result = _compare(make_snapshot(items=base), make_snapshot(items=current))

        by_sample = {i.key.source_sample_id: i for i in result.item_diffs}
        # The failed item is a status change, NOT an outcome (score) change.
        assert by_sample["s0"].status_changed is True
        assert by_sample["s0"].outcome_changed is False
        # Outcome drift only counts the 3 graded-in-both items — all unchanged.
        assert result.outcome_changed_count == 0
        assert result.outcome_changed_ratio == 0.0


# ===========================================================================
# Acceptance scenarios
# ===========================================================================


class TestAcceptanceScenarios:
    def test_case_a_stable(self) -> None:
        items = [
            {
                "task_id": "gsm8k_quick_v1",
                "source_sample_id": f"s{i}",
                "status": ItemStatus.GRADED,
                "score": 1.0,
                "response_text": "42",
            }
            for i in range(4)
        ]
        result = _compare(make_snapshot(items=items), make_snapshot(items=items))
        assert result.drift_level == BehaviorDriftLevel.NO_SIGNIFICANT_DRIFT

    def test_case_b_behavior_change_but_same_capability(self) -> None:
        base = [
            {
                "task_id": "gsm8k_quick_v1",
                "source_sample_id": "s0",
                "status": ItemStatus.GRADED,
                "score": 1.0,
                "response_text": "The answer is 42.",
            }
        ]
        current = [
            {
                "task_id": "gsm8k_quick_v1",
                "source_sample_id": "s0",
                "status": ItemStatus.GRADED,
                "score": 1.0,
                "response_text": "Therefore, after calculation, the answer is 42.",
            }
        ]
        result = _compare(make_snapshot(items=base), make_snapshot(items=current))
        assert result.output_changed_count == 1
        assert result.outcome_changed_count == 0
        assert result.drift_level != BehaviorDriftLevel.MATERIAL_DRIFT

    def test_case_c_real_result_change_is_material_and_traceable(self) -> None:
        base = _4_ITEMS
        current = [
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s0", "status": ItemStatus.GRADED, "score": 0.0},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s1", "status": ItemStatus.GRADED, "score": 0.0},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s2", "status": ItemStatus.GRADED, "score": 1.0},
            {"task_id": "gsm8k_quick_v1", "source_sample_id": "s3", "status": ItemStatus.GRADED, "score": 1.0},
        ]
        baseline = make_snapshot(items=base, profile=make_profile({CapabilityDimension.MATH_SCIENCE: 1.0}))
        current_snap = make_snapshot(items=current, profile=make_profile({CapabilityDimension.MATH_SCIENCE: 0.5}))
        result = _compare(baseline, current_snap)

        assert result.drift_level == BehaviorDriftLevel.MATERIAL_DRIFT
        # Traceability: every changed item carries evidence refs on both sides.
        for item in result.item_diffs:
            assert item.baseline_evidence_refs
            assert item.current_evidence_refs
        changed = [i for i in result.item_diffs if i.outcome_changed]
        assert len(changed) == 2
        for item in changed:
            assert item.key.input_sha256  # stable identity present
            assert item.baseline_score == 1.0
            assert item.current_score == 0.0
