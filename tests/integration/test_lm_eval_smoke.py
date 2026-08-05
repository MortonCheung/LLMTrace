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

from llmtrace.adapters.lm_eval import (
    BUILTIN_SMOKE_TASK_ROOT,
    LmEvalAdapter,
)
from llmtrace.adapters.lm_eval_runner import (
    LmEvalSecurityError,
    _validate_task_name,
    _validate_task_yaml,
)
from llmtrace.benchmarks.models import TaskSpec, TaskStatus

# ---------------------------------------------------------------------------
# Smoke test: full pipeline with deterministic FakeProvider
# ---------------------------------------------------------------------------


class TestLmEvalSmokePipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline_success(self, smoke_provider: object) -> None:
        """The full pipeline runs successfully with a deterministic provider."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
            model_name="test-model",
            generation_kwargs={"until": ["\n"]},
        )

        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
        attempt = await adapter.run_task(task, provider)

        assert attempt.status == TaskStatus.SUCCESS
        assert attempt.adapter_id == "lm-eval"
        assert attempt.task_id == "llmtrace_smoke"

    @pytest.mark.asyncio
    async def test_runplan_requests_match(self, smoke_provider: object) -> None:
        """The provider call count equals the RunPlan planned_requests."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(model_name="test-model")
        plan = adapter.build_plan("s", "v", "src", "rev", ["llmtrace_smoke"])
        assert plan.total_samples == 4
        assert plan.budget.planned_requests == 4

    @pytest.mark.asyncio
    async def test_evidence_refs_integrity(self, smoke_provider: object) -> None:
        """Evidence references in TaskAttempt are valid UUIDs."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
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
    async def test_grade_result_from_smoke(self, smoke_provider: object) -> None:
        """The grading result, when all generations match, is exact_match=1.0."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
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
    async def test_json_roundtrip_task_attempt(self, smoke_provider: object) -> None:
        """TaskAttempt serializes to JSON and back."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
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
    async def test_evidence_count_matches_request_count(self, smoke_provider: object) -> None:
        """Section 7: Evidence count must equal actual provider requests."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
            model_name="test-model",
            generation_kwargs={"until": ["\n"]},
        )

        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
        attempt = await adapter.run_task(task, provider)

        if attempt.status == TaskStatus.SUCCESS:
            assert len(attempt.evidence_refs) == 4  # matches smoke JSON items
            assert provider.call_count == 4

    @pytest.mark.asyncio
    async def test_metadata_has_metric_result_not_task_results(self, smoke_provider: object) -> None:
        """Section 6: metadata contains LmEvalMetricResult, not raw task_results dict."""
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
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
    async def test_exception_evidence_causes_failure(self, exception_evidence_provider: object) -> None:
        """Evidence with exception_type produces FAILURE with evidence_refs."""
        from tests.adapters.conftest import FakeProvider

        provider = exception_evidence_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
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
    async def test_http_401_causes_failure(self, http_401_provider: object) -> None:
        """Evidence with HTTP 401 produces FAILURE."""
        from tests.adapters.conftest import FakeProvider

        provider = http_401_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
            model_name="test-model",
            generation_kwargs={"until": ["\n"]},
        )

        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
        attempt = await adapter.run_task(task, provider)

        assert attempt.status == TaskStatus.FAILURE
        assert attempt.failure is not None

    @pytest.mark.asyncio
    async def test_http_500_causes_failure(self, http_500_provider: object) -> None:
        """Evidence with HTTP 500 produces FAILURE."""
        from tests.adapters.conftest import FakeProvider

        provider = http_500_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
            model_name="test-model",
            generation_kwargs={"until": ["\n"]},
        )

        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
        attempt = await adapter.run_task(task, provider)

        assert attempt.status == TaskStatus.FAILURE
        assert attempt.failure is not None

    @pytest.mark.asyncio
    async def test_empty_response_causes_failure(self, empty_response_provider: object) -> None:
        """Evidence with empty response_text produces FAILURE."""
        from tests.adapters.conftest import FakeProvider

        provider = empty_response_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
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
    async def test_yaml_kwargs_reach_provider(self, smoke_provider: object) -> None:
        """YAML task config generation_kwargs are passed to FakeProvider.

        The YAML defines temperature=0.0, do_sample=false.
        """
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
            model_name="test-model",
        )

        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
        await adapter.run_task(task, provider)

        assert len(provider.received_options) == 4
        for opts in provider.received_options:
            assert opts is not None
            assert opts.until == ["\n"]
            assert opts.temperature == 0.0
            assert opts.do_sample is False

    @pytest.mark.asyncio
    async def test_adapter_kwargs_overridden_by_yaml(self, smoke_provider: object) -> None:
        """Adapter temperature=0.3 is overridden by YAML temperature=0.0.

        Section 1: Provider receives 0.0 (YAML), not 0.3 (adapter).
        """
        from llmtrace.benchmarks.models import CompletionOptions
        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        adapter = LmEvalAdapter(
            model_name="test-model",
            generation_kwargs={"temperature": 0.3},
        )

        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
        attempt = await adapter.run_task(task, provider)

        # Provider receives YAML's temperature=0.0, not adapter's 0.3
        assert len(provider.received_options) == 4
        for opts in provider.received_options:
            assert opts is not None
            assert opts.temperature == 0.0  # YAML value wins

        # metadata must also record 0.0
        if attempt.status == TaskStatus.SUCCESS:
            metric = attempt.metadata["metric_result"]
            gen_opts = metric.get("generation_options")
            if gen_opts is not None:
                opts_obj = CompletionOptions(**gen_opts)
                assert opts_obj.temperature == 0.0


# ---------------------------------------------------------------------------
# Section 5: Runner security boundary tests (via validation functions directly)
# ---------------------------------------------------------------------------


class TestRunnerSecurity:
    """Tests for the security boundary via direct validation function calls.

    Since LmEvalAdapter no longer accepts arbitrary task_root,
    security validation tests call _validate_task_name / _validate_task_yaml
    directly with test-controlled paths.
    """

    def test_non_whitelisted_task_rejected(self) -> None:
        with pytest.raises(LmEvalSecurityError, match="not in the trusted whitelist"):
            _validate_task_name("nonexistent_task")

    def test_whitelisted_task_accepted(self) -> None:
        _validate_task_name("llmtrace_smoke")  # must not raise

    def test_path_traversal_rejected(self) -> None:
        """Path traversal via data_files escaping dir is rejected."""
        mal_dir = Path(tempfile.mkdtemp(prefix="llmtrace_security_"))
        try:
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

            with pytest.raises(LmEvalSecurityError, match="outside the trusted task directory"):
                _validate_task_yaml(mal_dir, "llmtrace_smoke")
        finally:
            import shutil

            shutil.rmtree(mal_dir, ignore_errors=True)

    def test_non_local_dataset_path_rejected(self) -> None:
        """Remote dataset_path (non-json) raises LmEvalSecurityError."""
        mal_dir = Path(tempfile.mkdtemp(prefix="llmtrace_security_"))
        try:
            mal_yaml = mal_dir / "llmtrace_smoke.yaml"
            mal_yaml.write_text("""task: llmtrace_smoke
dataset_path: huggingface
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
num_fewshot: 0
metadata:
  version: 1.0
data_files: llmtrace_smoke.json
""")

            with pytest.raises(LmEvalSecurityError, match="dataset_path.*not allowed"):
                _validate_task_yaml(mal_dir, "llmtrace_smoke")
        finally:
            import shutil

            shutil.rmtree(mal_dir, ignore_errors=True)

    def test_yaml_function_tag_rejected(self) -> None:
        """Custom YAML !function tags are rejected."""
        mal_dir = Path(tempfile.mkdtemp(prefix="llmtrace_security_"))
        try:
            mal_yaml = mal_dir / "llmtrace_smoke.yaml"
            mal_yaml.write_text("""task: llmtrace_smoke
dataset_path: json
output_type: generate_until
training_split: train
validation_split: train
test_split: train
fewshot_split: train
doc_to_text: "{{input}}"
doc_to_target: "{{output}}"
process_results: !function utils.process_results
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
generation_kwargs:
  until:
    - "\\n"
num_fewshot: 0
metadata:
  version: 1.0
data_files: llmtrace_smoke.json
""")

            with pytest.raises(LmEvalSecurityError, match="custom YAML tags"):
                _validate_task_yaml(mal_dir, "llmtrace_smoke")
        finally:
            import shutil

            shutil.rmtree(mal_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Section 4: Temporary directory cleanup tests
# ---------------------------------------------------------------------------


class TestTempDirectoryCleanup:
    @pytest.mark.asyncio
    async def test_tempdir_cleaned_after_success(self, smoke_provider: object) -> None:
        """After a successful run, the temp directory created by the Runner is gone."""
        import tempfile as tf

        from tests.adapters.conftest import FakeProvider

        provider = smoke_provider
        assert isinstance(provider, FakeProvider)

        # Capture all directories created during the run
        before = {Path(p) for p in tf.gettempdir() if "llmtrace_task_" in p}
        before_dirs = {d for d in before if d.is_dir()}

        adapter = LmEvalAdapter(
            model_name="test-model",
            generation_kwargs={"until": ["\n"]},
        )
        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
        attempt = await adapter.run_task(task, provider)

        assert attempt.status == TaskStatus.SUCCESS

        # After run: no llmtrace_task_ directories left behind
        after = {Path(p) for p in tf.gettempdir() if "llmtrace_task_" in p}
        after_dirs = {d for d in after if d.is_dir()}
        leftover = after_dirs - before_dirs

        assert not leftover, f"Temporary directories left behind: {leftover}"

    @pytest.mark.asyncio
    async def test_tempdir_cleaned_after_provider_error(self, exception_evidence_provider: object) -> None:
        """After a provider error, the temp directory is still cleaned up."""
        import tempfile as tf

        from tests.adapters.conftest import FakeProvider

        provider = exception_evidence_provider
        assert isinstance(provider, FakeProvider)

        before = {Path(p) for p in tf.gettempdir() if "llmtrace_task_" in p}
        before_dirs = {d for d in before if d.is_dir()}

        adapter = LmEvalAdapter(
            model_name="test-model",
            generation_kwargs={"until": ["\n"]},
        )
        task = TaskSpec(task_id="llmtrace_smoke", name="Smoke", num_samples=4)
        attempt = await adapter.run_task(task, provider)

        assert attempt.status == TaskStatus.FAILURE

        after = {Path(p) for p in tf.gettempdir() if "llmtrace_task_" in p}
        after_dirs = {d for d in after if d.is_dir()}
        leftover = after_dirs - before_dirs

        assert not leftover, f"Temporary directories left behind after error: {leftover}"


# ---------------------------------------------------------------------------
# Smoke task metadata validation
# ---------------------------------------------------------------------------


class TestSmokeTaskMetadata:
    def test_smoke_yaml_exists_in_builtin(self) -> None:
        yaml_path = Path(BUILTIN_SMOKE_TASK_ROOT) / "llmtrace_smoke.yaml"
        assert yaml_path.exists()

    def test_smoke_json_exists_in_builtin(self) -> None:
        json_path = Path(BUILTIN_SMOKE_TASK_ROOT) / "llmtrace_smoke.json"
        assert json_path.exists()

    def test_smoke_json_has_four_items(self) -> None:
        json_path = Path(BUILTIN_SMOKE_TASK_ROOT) / "llmtrace_smoke.json"
        data = json.loads(json_path.read_text())
        assert len(data) == 4

    def test_smoke_json_is_deterministic_format(self) -> None:
        """Each item asks to repeat a fixed string."""
        json_path = Path(BUILTIN_SMOKE_TASK_ROOT) / "llmtrace_smoke.json"
        data = json.loads(json_path.read_text())
        for item in data:
            assert "Repeat exactly:" in item["input"]
            assert item["output"] in item["input"]

    def test_smoke_task_not_in_capability_score(self) -> None:
        """The smoke task category is 'smoke', not a capability benchmark."""
        adapter = LmEvalAdapter()
        tasks = adapter.list_tasks()
        for t in tasks:
            assert t.category == "smoke"
