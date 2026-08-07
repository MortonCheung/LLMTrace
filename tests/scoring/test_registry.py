"""Tests for TaskScoringRegistry."""

from __future__ import annotations

import pytest

from llmtrace.scoring.aggregator import TaskScoringRegistry
from llmtrace.scoring.errors import TaskRegistrationError
from llmtrace.scoring.models import CapabilityDimension, TaskScoringSpec


class TestRegistryConstruction:
    """Test registry construction."""

    def test_empty_registry(self) -> None:
        registry = TaskScoringRegistry()
        assert len(registry) == 0

    def test_registry_with_single_spec(self) -> None:
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        assert len(registry) == 1
        assert "task_a" in registry

    def test_duplicate_task_id_raises(self) -> None:
        with pytest.raises(TaskRegistrationError, match="Duplicate"):
            TaskScoringRegistry(
                [
                    TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
                    TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
                ]
            )

    def test_contains(self) -> None:
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        assert "task_a" in registry
        assert "missing" not in registry

    def test_get_returns_none_for_missing(self) -> None:
        registry = TaskScoringRegistry()
        assert registry.get("nonexistent") is None

    def test_get_returns_deep_copy(self) -> None:
        """get() must return a deep copy — mutations do not affect registry."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        spec = registry.get("task_a")
        assert spec is not None
        # spec is frozen, so we can't mutate it — which is the point
        with pytest.raises((TypeError, ValueError)):
            spec.task_weight = 999  # type: ignore[misc]

        # Original in registry unchanged
        original = registry.get("task_a")
        assert original is not None
        assert original.task_weight == 1.0

    def test_iteration_returns_deep_copies(self) -> None:
        """Iteration must yield deep copies — mutations do not affect registry."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        for spec in registry:
            # spec is frozen, can't mutate it
            with pytest.raises((TypeError, ValueError)):
                spec.task_weight = 999  # type: ignore[misc]

    def test_items_returns_deep_copies(self) -> None:
        """items() must yield deep copies."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        for _task_id, spec in registry.items():
            # spec is frozen, can't mutate
            with pytest.raises((TypeError, ValueError)):
                spec.task_weight = 999  # type: ignore[misc]

    def test_registry_immutable_after_construction(self) -> None:
        """Registry cannot be modified after construction (no set/del)."""
        registry = TaskScoringRegistry(
            [
                TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0),
            ]
        )
        # Cannot set item
        with pytest.raises(TypeError):
            registry["task_b"] = TaskScoringSpec(  # type: ignore[index]
                task_id="task_b",
                dimension=CapabilityDimension.CODING,
                task_weight=1.0,
            )
        # Cannot delete item
        with pytest.raises(TypeError):
            del registry["task_a"]  # type: ignore[arg-type]

    def test_task_weight_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=0.0)

    def test_task_weight_negative_fails(self) -> None:
        with pytest.raises(ValueError):
            TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=-1.0)

    def test_default_capability_score_eligible(self) -> None:
        spec = TaskScoringSpec(task_id="task_a", dimension=CapabilityDimension.REASONING, task_weight=1.0)
        assert spec.capability_score_eligible is True
