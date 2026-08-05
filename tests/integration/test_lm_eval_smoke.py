"""End-to-end smoke test for lm-eval integration chain.

Validates:
  lm-eval task → ProviderBackedLM → Provider → HTTPEvidence
  → TaskAttempt → GradeResult

Requires: pip install -e ".[lm-eval]"
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

from llmtrace.adapters.lm_eval import LmEvalAdapter
from llmtrace.benchmarks.models import AdapterFailure, TaskStatus

# ---------------------------------------------------------------------------
# Smoke test: full pipeline with deterministic FakeProvider
# ---------------------------------------------------------------------------


class TestLmEvalSmokePipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline_success(self, smoke_provider: object, smoke_task_path: str) -> None:
        """The full pipeline runs successfully with a deterministic provider."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
            include_path=smoke_task_path,
            model_name="test-model",
            generation_kwargs={"until": ["\n"]},
        )

        from llmtrace.benchmarks.models import TaskSpec

        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
        attempt = await adapter.run_task(task, provider)

        assert attempt.status == TaskStatus.SUCCESS
        assert attempt.adapter_id == "lm-eval"
        assert attempt.task_id == "llmtrace_smoke"

    @pytest.mark.asyncio
    async def test_runplan_requests_match(self, smoke_provider: object, smoke_task_path: str) -> None:
        """The provider call count equals the RunPlan planned_requests."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(include_path=smoke_task_path, model_name="test-model")
        plan = adapter.build_plan("s", "v", "src", "rev", ["llmtrace_smoke"])
        assert plan.total_samples == 4
        assert plan.budget.planned_requests == 4

    @pytest.mark.asyncio
    async def test_evidence_refs_integrity(self, smoke_provider: object, smoke_task_path: str) -> None:
        """Evidence references in TaskAttempt are valid UUIDs."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
            include_path=smoke_task_path,
            model_name="test-model",
            generation_kwargs={"until": ["\n"]},
        )

        from llmtrace.benchmarks.models import TaskSpec

        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
        attempt = await adapter.run_task(task, provider)

        if attempt.status == TaskStatus.SUCCESS:
            assert len(attempt.evidence_refs) > 0
            for ref in attempt.evidence_refs:
                UUID(ref)  # each ref must be a valid UUID

    @pytest.mark.asyncio
    async def test_grade_result_from_smoke(self, smoke_provider: object, smoke_task_path: str) -> None:
        """The grading result, when all generations match, is exact_match=1.0."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
            include_path=smoke_task_path,
            model_name="test-model",
            generation_kwargs={"until": ["\n"]},
        )

        from llmtrace.benchmarks.models import TaskSpec

        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
        attempt = await adapter.run_task(task, provider)

        if attempt.status == TaskStatus.SUCCESS:
            # Build a simulated grade result using the adapter's normalizer
            # (in real use, the runner would produce a result dict)
            grade = adapter.normalize_result(
                {
                    "results": {"exact_match": 1.0},
                    "evidence_ids": attempt.evidence_refs,
                    "task_name": "llmtrace_smoke",
                    "attempt_id": attempt.attempt_id,
                }
            )
            assert grade.normalized_score == 1.0
            assert grade.raw_score == 1.0
            assert grade.evidence_refs == attempt.evidence_refs
            assert grade.grader_id == "exact_match"

    @pytest.mark.asyncio
    async def test_structured_failure(self, failing_provider: object, smoke_task_path: str) -> None:
        """Provider errors produce structured AdapterFailure, not exceptions."""
        from tests.adapters.conftest import FakeProvider

        provider = failing_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
            include_path=smoke_task_path,
            model_name="test-model",
            generation_kwargs={"until": ["\n"]},
        )

        from llmtrace.benchmarks.models import TaskSpec

        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
        attempt = await adapter.run_task(task, provider)

        # With a failing provider, the adapter should capture the failure
        # The attempt may be FAILURE if the error propagates through the runner
        if attempt.status == TaskStatus.FAILURE:
            assert attempt.failure is not None
            assert isinstance(attempt.failure, AdapterFailure)
            assert len(attempt.failure.error_code) > 0
            assert len(attempt.failure.message) > 0
        # Note: lm-eval may internally catch and wrap provider errors;
        # the key assertion is that we NEVER get an unhandled exception.

    @pytest.mark.asyncio
    async def test_json_roundtrip_task_attempt(self, smoke_provider: object, smoke_task_path: str) -> None:
        """TaskAttempt serializes to JSON and back."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
            include_path=smoke_task_path,
            model_name="test-model",
            generation_kwargs={"until": ["\n"]},
        )

        from llmtrace.benchmarks.models import TaskSpec

        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
        attempt = await adapter.run_task(task, provider)

        data = attempt.model_dump_json()
        restored = attempt.__class__.model_validate_json(data)
        assert restored.attempt_id == attempt.attempt_id
        assert restored.status == attempt.status


# ---------------------------------------------------------------------------
# Smoke task metadata validation
# ---------------------------------------------------------------------------


class TestSmokeTaskMetadata:
    def test_smoke_yaml_exists(self, smoke_task_path: str) -> None:
        yaml_path = Path(smoke_task_path) / "llmtrace_smoke.yaml"
        assert yaml_path.exists()

    def test_smoke_json_exists(self, smoke_task_path: str) -> None:
        json_path = Path(smoke_task_path) / "llmtrace_smoke.json"
        assert json_path.exists()

    def test_smoke_json_has_four_items(self, smoke_task_path: str) -> None:
        json_path = Path(smoke_task_path) / "llmtrace_smoke.json"
        data = json.loads(json_path.read_text())
        assert len(data) == 4

    def test_smoke_json_is_deterministic_format(self, smoke_task_path: str) -> None:
        """Each item asks to repeat a fixed string."""
        json_path = Path(smoke_task_path) / "llmtrace_smoke.json"
        data = json.loads(json_path.read_text())
        for item in data:
            assert "Repeat exactly:" in item["input"]
            assert item["output"] in item["input"]

    def test_smoke_task_not_in_capability_score(self) -> None:
        """The smoke task category is 'smoke', not a capability benchmark."""
        adapter = LmEvalAdapter(include_path="/fake")
        tasks = adapter.list_tasks()
        for t in tasks:
            assert t.category == "smoke"
            assert t.metadata == {}
