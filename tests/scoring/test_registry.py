"""Tests for TaskScoringRegistry."""

from __future__ import annotations

import pytest

from llmtrace.scoring.aggregator import TaskScoringRegistry
from llmtrace.scoring.errors import TaskRegistrationError
from llmtrace.scoring.models import CapabilityDimension, TaskScoringSpec


class TestRegistryCreation:
    """Test registry construction and basic operations."""

    def test_empty_registry(self) -> None:
        registry = TaskScoringRegistry()
        assert len(registry) == 0

    def test_register_single_spec(self) -> None:
        spec = TaskScoringSpec(
            task_id="mmlu_abstract_algebra",
            dimension=CapabilityDimension.REASONING,
            task_weight=1.0,
        )
        registry = TaskScoringRegistry([spec])
        assert len(registry) == 1
        assert registry.get("mmlu_abstract_algebra") is not None

    def test_duplicate_task_id_fails(self) -> None:
        spec1 = TaskScoringSpec(
            task_id="dup_task",
            dimension=CapabilityDimension.REASONING,
            task_weight=1.0,
        )
        spec2 = TaskScoringSpec(
            task_id="dup_task",
            dimension=CapabilityDimension.CODING,
            task_weight=1.0,
        )
        with pytest.raises(TaskRegistrationError, match="Duplicate task_id"):
            TaskScoringRegistry([spec1, spec2])

    def test_get_unregistered_returns_none(self) -> None:
        registry = TaskScoringRegistry()
        assert registry.get("nonexistent") is None

    def test_contains(self) -> None:
        spec = TaskScoringSpec(
            task_id="test_task",
            dimension=CapabilityDimension.REASONING,
            task_weight=1.0,
        )
        registry = TaskScoringRegistry([spec])
        assert "test_task" in registry
        assert "other" not in registry

    def test_iteration(self) -> None:
        specs = [
            TaskScoringSpec(task_id="t1", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            TaskScoringSpec(task_id="t2", dimension=CapabilityDimension.CODING, task_weight=1.0),
        ]
        registry = TaskScoringRegistry(specs)
        items = list(registry)
        assert len(items) == 2

    def test_deep_copy_semantics(self) -> None:
        """Modifying the original spec list must not affect the registry."""
        spec = TaskScoringSpec(
            task_id="test_task",
            dimension=CapabilityDimension.REASONING,
            task_weight=1.0,
        )
        registry = TaskScoringRegistry([spec])
        # Modify original
        spec.task_weight = 99.0
        # Registry copy unchanged
        stored = registry.get("test_task")
        assert stored is not None
        assert stored.task_weight == 1.0

    def test_task_weight_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            TaskScoringSpec(
                task_id="zero_weight",
                dimension=CapabilityDimension.REASONING,
                task_weight=0.0,
            )

    def test_task_weight_negative_fails(self) -> None:
        with pytest.raises(ValueError):
            TaskScoringSpec(
                task_id="neg_weight",
                dimension=CapabilityDimension.REASONING,
                task_weight=-1.0,
            )

    def test_capability_score_eligible_default_true(self) -> None:
        spec = TaskScoringSpec(
            task_id="default_eligible",
            dimension=CapabilityDimension.REASONING,
            task_weight=1.0,
        )
        assert spec.capability_score_eligible is True
