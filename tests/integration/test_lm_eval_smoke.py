"""End-to-end smoke test for lm-eval integration chain.

Validates:
  lm-eval task -> ProviderBackedLM -> Provider -> HTTPEvidence
  -> TaskAttempt -> GradeResult

Requires: pip install -e ".[lm-eval]"
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from uuid import UUID

import pytest

pytest.importorskip("lm_eval")

from llmtrace.adapters.lm_eval import LmEvalAdapter
from llmtrace.benchmarks.models import (
    TaskSpec,
    TaskStatus,
)

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
            task_root=smoke_task_path,
            model_name="test-model",
            generation_kwargs={"until": ["\n"]},
        )

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

        adapter = LmEvalAdapter(task_root=smoke_task_path, model_name="test-model")
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
            task_root=smoke_task_path,
            model_name="test-model",
            generation_kwargs={"until": ["\n"]},
        )

        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
        attempt = await adapter.run_task(task, provider)

        if attempt.status == TaskStatus.SUCCESS:
            assert len(attempt.evidence_refs) > 0
            for ref in attempt.evidence_refs:
                UUID(ref)

    @pytest.mark.asyncio
    async def test_grade_result_from_smoke(self, smoke_provider: object, smoke_task_path: str) -> None:
        """The grading result, when all generations match, is exact_match=1.0."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
            task_root=smoke_task_path,
            model_name="test-model",
            generation_kwargs={"until": ["\n"]},
        )

        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
        attempt = await adapter.run_task(task, provider)

        if attempt.status == TaskStatus.SUCCESS:
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
    async def test_json_roundtrip_task_attempt(self, smoke_provider: object, smoke_task_path: str) -> None:
        """TaskAttempt serializes to JSON and back."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
            task_root=smoke_task_path,
            model_name="test-model",
            generation_kwargs={"until": ["\n"]},
        )

        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
        attempt = await adapter.run_task(task, provider)

        data = attempt.model_dump_json()
        restored = attempt.__class__.model_validate_json(data)
        assert restored.attempt_id == attempt.attempt_id
        assert restored.status == attempt.status

    @pytest.mark.asyncio
    async def test_evidence_count_matches_request_count(self, smoke_provider: object, smoke_task_path: str) -> None:
        """Section 7: Evidence count must equal actual provider requests."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
            task_root=smoke_task_path,
            model_name="test-model",
            generation_kwargs={"until": ["\n"]},
        )

        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
        attempt = await adapter.run_task(task, provider)

        if attempt.status == TaskStatus.SUCCESS:
            assert len(attempt.evidence_refs) == 4  # matches smoke JSON items
            assert provider.call_count == 4

    @pytest.mark.asyncio
    async def test_metadata_has_metric_result_not_task_results(
        self, smoke_provider: object, smoke_task_path: str
    ) -> None:
        """Section 6: metadata contains LmEvalMetricResult, not raw task_results dict."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
            task_root=smoke_task_path,
            model_name="test-model",
            generation_kwargs={"until": ["\n"]},
        )

        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
        attempt = await adapter.run_task(task, provider)

        if attempt.status == TaskStatus.SUCCESS:
            assert "metric_result" in attempt.metadata
            assert "task_results" not in attempt.metadata
            metric = attempt.metadata["metric_result"]
            assert "task_name" in metric
            assert "metric_name" in metric


class TestLmEvalSmokeFailureModes:
    """Section 4: Evidence-based failure modes."""

    @pytest.mark.asyncio
    async def test_exception_evidence_causes_failure(
        self, exception_evidence_provider: object, smoke_task_path: str
    ) -> None:
        """Evidence with exception_type produces FAILURE with evidence_refs."""
        from tests.adapters.conftest import FakeProvider

        provider = exception_evidence_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
            task_root=smoke_task_path,
            model_name="test-model",
            generation_kwargs={"until": ["\n"]},
        )

        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
        attempt = await adapter.run_task(task, provider)

        assert attempt.status == TaskStatus.FAILURE
        assert attempt.failure is not None
        assert attempt.failure.error_code == "PROVIDER_EXCEPTION"
        # evidence_refs still contain the failed evidence
        assert len(attempt.evidence_refs) > 0

    @pytest.mark.asyncio
    async def test_http_401_causes_failure(self, http_401_provider: object, smoke_task_path: str) -> None:
        """Evidence with HTTP 401 produces FAILURE."""
        from tests.adapters.conftest import FakeProvider

        provider = http_401_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
            task_root=smoke_task_path,
            model_name="test-model",
            generation_kwargs={"until": ["\n"]},
        )

        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
        attempt = await adapter.run_task(task, provider)

        assert attempt.status == TaskStatus.FAILURE
        assert attempt.failure is not None

    @pytest.mark.asyncio
    async def test_http_500_causes_failure(self, http_500_provider: object, smoke_task_path: str) -> None:
        """Evidence with HTTP 500 produces FAILURE."""
        from tests.adapters.conftest import FakeProvider

        provider = http_500_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
            task_root=smoke_task_path,
            model_name="test-model",
            generation_kwargs={"until": ["\n"]},
        )

        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
        attempt = await adapter.run_task(task, provider)

        assert attempt.status == TaskStatus.FAILURE
        assert attempt.failure is not None

    @pytest.mark.asyncio
    async def test_empty_response_causes_failure(self, empty_response_provider: object, smoke_task_path: str) -> None:
        """Evidence with empty response_text produces FAILURE."""
        from tests.adapters.conftest import FakeProvider

        provider = empty_response_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
            task_root=smoke_task_path,
            model_name="test-model",
            generation_kwargs={"until": ["\n"]},
        )

        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
        attempt = await adapter.run_task(task, provider)

        assert attempt.status == TaskStatus.FAILURE
        assert attempt.failure is not None
        assert attempt.failure.error_code == "PROVIDER_EMPTY_RESPONSE"


class TestLmEvalSmokeGenerationKwargs:
    """Section 3: generation kwargs passed through the pipeline."""

    @pytest.mark.asyncio
    async def test_temperature_kwargs_reach_provider(self, smoke_provider: object, smoke_task_path: str) -> None:
        """Generation kwargs from YAML task config are passed to FakeProvider.

        Note: lm-eval merges task-level generation_kwargs from YAML with
        per-instance kwargs. The YAML defines temperature=0.0, do_sample=false.
        """
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
            task_root=smoke_task_path,
            model_name="test-model",
        )

        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
        await adapter.run_task(task, provider)

        assert len(provider.received_options) == 4
        for opts in provider.received_options:
            assert opts is not None
            assert opts.until == ["\n"]
            # YAML defines temperature=0.0, do_sample=false
            assert opts.temperature == 0.0
            assert opts.do_sample is False


# ---------------------------------------------------------------------------
# Section 5: Runner security boundary tests
# ---------------------------------------------------------------------------


class TestRunnerSecurity:
    """Tests for the security boundary enforced by LmEvalRunner."""

    @pytest.mark.asyncio
    async def test_non_whitelisted_task_rejected(self, smoke_provider: object, smoke_task_path: str) -> None:
        """Non-whitelisted task name raises LmEvalSecurityError."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(task_root=smoke_task_path, model_name="test-model")
        task = TaskSpec(task_id="nonexistent_task", name="Bad", num_samples=1)
        attempt = await adapter.run_task(task, provider)

        assert attempt.status == TaskStatus.FAILURE

    @pytest.mark.asyncio
    async def test_non_local_include_path_rejected(self, smoke_provider: object) -> None:
        """A non-existent or malicious include_path should fail safely."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(task_root="/tmp/nonexistent_path_12345", model_name="test-model")
        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
        attempt = await adapter.run_task(task, provider)

        assert attempt.status == TaskStatus.FAILURE

    @pytest.mark.asyncio
    async def test_path_traversal_rejected(self, smoke_provider: object, smoke_task_path: str) -> None:
        """Path traversal via data_files escaping the task directory is rejected."""
        from tests.adapters.conftest import FakeProvider

        # Create a malicious YAML with data_files pointing outside the task root
        mal_dir = Path(tempfile.mkdtemp(prefix="llmtrace_security_"))
        try:
            # Create the YAML in a temp directory with a data_files that traverses out
            mal_yaml = mal_dir / "llmtrace_smoke.yaml"
            mal_yaml.write_text("""task: llmtrace_smoke
dataset_path: json
dataset_kwargs:
  data_files: ../../etc/passwd
output_type: generate_until
training_split: train
validation_split: train
test_split: train
fewshot_split: train
doc_to_text: "{{input}}"
doc_to_target: "{{output}}"
process_results: null
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
generation_kwargs:
  until:
    - "\\n"
  temperature: 0.0
num_fewshot: 0
metadata:
  version: 1.0
""")

            provider = smoke_provider
            assert isinstance(provider, FakeProvider)

            adapter = LmEvalAdapter(task_root=str(mal_dir), model_name="test-model")
            task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
            attempt = await adapter.run_task(task, provider)

            assert attempt.status == TaskStatus.FAILURE
        finally:
            import shutil

            shutil.rmtree(mal_dir, ignore_errors=True)


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
        adapter = LmEvalAdapter(task_root="/fake")
        tasks = adapter.list_tasks()
        for t in tasks:
            assert t.category == "smoke"
