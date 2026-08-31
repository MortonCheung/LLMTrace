"""Shared helpers for the v0.4-A reference-layer tests.

These helpers build *valid* Quick Suite run artifacts (manifest +
``benchmark_runs.json`` + ``capability_profile.json``) that pass every
qualification gate by default, and let individual tests corrupt one gate at
a time to assert a REJECT with the expected machine-readable reason code.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from llmtrace.adapters.quick_suite import (
    QUICK_SUITE_BENCHMARK_REQUESTS,
    QUICK_SUITE_SUITE_ID,
    QUICK_SUITE_SUITE_VERSION,
    get_quick_suite_content_sha256,
    get_quick_suite_generation_config,
)
from llmtrace.benchmarks.models import (
    AdapterFailure,
    BenchmarkItemResult,
    BenchmarkRunResult,
    FailureCategory,
    ItemStatus,
    TaskAttempt,
    TaskStatus,
)
from llmtrace.execution.artifacts import RunArtifactRepository
from llmtrace.execution.models import RunArtifactManifest, UnifiedRunStatus
from llmtrace.scoring.models import (
    CapabilityDimension,
    CapabilityProfile,
    DimensionScoreResult,
    DimensionScoreStatus,
)
from llmtrace.scoring.reference import ReferenceProvenance, ReferenceSnapshot

DEFAULT_EXECUTION_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

# (task_id, source_id, dimension) — the four Quick Suite tasks.
_QUICK_TASKS: tuple[tuple[str, str, str], ...] = (
    ("arc_challenge_quick_v1", "arc_challenge", "reasoning"),
    ("humaneval_quick_v1", "humaneval", "coding"),
    ("gsm8k_quick_v1", "gsm8k", "math_science"),
    ("ifeval_quick_v1", "ifeval", "instruction_following"),
)

_INPUT_SHA256 = "a" * 64
_PROVIDER_ID = "openai"
_ADAPTER_ID = "llmtrace-quick-v1"
_ADAPTER_VERSION = "0.1.0"
_SCORING_POLICY_ID = "llmtrace-capability-v1"
_SCORING_POLICY_VERSION = "0.1.0"
_QUALIFICATION_POLICY_ID = "llmtrace_reference_qualification_v1"
_QUALIFICATION_POLICY_VERSION = "0.1.0"
_COVERAGE_WEIGHT = 0.75  # sum of the four enabled dimension weights


def expected_generation_config_sha256() -> str:
    """Canonical SHA-256 of the Quick Suite generation config — mirrors
    ``qualification._expected_generation_config_sha256``."""
    config = get_quick_suite_generation_config()
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Run artifact construction
# ---------------------------------------------------------------------------


_MISSING = object()


def make_manifest(
    execution_id: str = DEFAULT_EXECUTION_ID,
    *,
    status: UnifiedRunStatus = UnifiedRunStatus.COMPLETED,
    suite_id: str = QUICK_SUITE_SUITE_ID,
    suite_version: str = QUICK_SUITE_SUITE_VERSION,
    suite_content_sha256: str | None | object = _MISSING,
    generation_config_sha256: str | None = None,
    scoring_policy_id: str = _SCORING_POLICY_ID,
    scoring_policy_version: str = _SCORING_POLICY_VERSION,
    adapter_id: str = _ADAPTER_ID,
    adapter_version: str = _ADAPTER_VERSION,
    completed_at: datetime | None = None,
) -> RunArtifactManifest:
    """A manifest that qualifies by default; override a field to corrupt a gate.

    ``suite_content_sha256`` defaults to the real Quick Suite content hash;
    pass ``None`` explicitly to model a pre-v0.4-A run that recorded no
    content identity.
    """
    if suite_content_sha256 is _MISSING:
        suite_content_sha256 = get_quick_suite_content_sha256()
    if generation_config_sha256 is None:
        generation_config_sha256 = expected_generation_config_sha256()
    return RunArtifactManifest(
        execution_id=execution_id,
        report_id="llmtrace_test",
        target_id="openai-test",
        candidate_model_id="my-real-model",
        base_url_redacted="http://test.example.com/v1",
        protocol="openai",
        created_at=datetime.now(UTC),
        completed_at=completed_at or datetime.now(UTC),
        status=status,
        suite_id=suite_id,
        suite_version=suite_version,
        suite_content_sha256=suite_content_sha256,
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        scoring_policy_id=scoring_policy_id,
        scoring_policy_version=scoring_policy_version,
        generation_config_sha256=generation_config_sha256,
        planned_requests=QUICK_SUITE_BENCHMARK_REQUESTS + 6,
        actual_requests=QUICK_SUITE_BENCHMARK_REQUESTS + 6,
    )


def _item(
    task_id: str,
    attempt_id: str,
    index: int,
    status: ItemStatus,
) -> BenchmarkItemResult:
    """One BenchmarkItemResult; GRADED items carry score 0.5 (a valid
    measurement — a wrong answer is still a measured answer)."""
    item_id = f"{task_id}:item-{index:03d}"
    if status == ItemStatus.GRADED:
        return BenchmarkItemResult(
            item_id=item_id,
            task_id=task_id,
            attempt_id=attempt_id,
            source_sample_id=f"sample-{task_id}-{index}",
            input_sha256=_INPUT_SHA256,
            status=status,
            raw_score=0.5,
            normalized_score=0.5,
            grader_id="quick-suite",
        )
    if status == ItemStatus.FAILURE:
        return BenchmarkItemResult(
            item_id=item_id,
            task_id=task_id,
            attempt_id=attempt_id,
            source_sample_id=f"sample-{task_id}-{index}",
            input_sha256=_INPUT_SHA256,
            status=status,
            raw_score=0.0,
            normalized_score=0.0,
            failure=AdapterFailure(
                error_code="HTTP_500",
                category=FailureCategory.PROVIDER,
                message="synthetic provider failure",
                retryable=False,
            ),
            grader_id="quick-suite",
        )
    return BenchmarkItemResult(
        item_id=item_id,
        task_id=task_id,
        attempt_id=attempt_id,
        source_sample_id=f"sample-{task_id}-{index}",
        input_sha256=_INPUT_SHA256,
        status=status,
        raw_score=0.0,
        normalized_score=0.0,
        error_message="synthetic ungradable answer",
        grader_id="quick-suite",
    )


def make_benchmark_runs(*, failure: int = 0, ungradable: int = 0) -> list[BenchmarkRunResult]:
    """32 item results across the four Quick Suite tasks (8 per task).

    The first ``failure`` items of the first task are FAILURE, the next
    ``ungradable`` are UNGRADABLE, everything else is GRADED.
    """
    if failure + ungradable > 8:
        raise ValueError("only the first task's 8 items can be corrupted per-task")
    runs: list[BenchmarkRunResult] = []
    for task_id, source_id, _dimension in _QUICK_TASKS:
        attempt_id = f"attempt-{task_id}"
        items: list[BenchmarkItemResult] = []
        for index in range(8):
            if task_id == _QUICK_TASKS[0][0] and index < failure:
                items.append(_item(task_id, attempt_id, index, ItemStatus.FAILURE))
            elif task_id == _QUICK_TASKS[0][0] and index < failure + ungradable:
                items.append(_item(task_id, attempt_id, index, ItemStatus.UNGRADABLE))
            else:
                items.append(_item(task_id, attempt_id, index, ItemStatus.GRADED))
        attempt = TaskAttempt(
            attempt_id=attempt_id,
            task_id=task_id,
            status=TaskStatus.SUCCESS,
            evidence_refs=[],
            item_results=items,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            source_id=source_id,
            source_revision="test-revision",
            suite_id=QUICK_SUITE_SUITE_ID,
            suite_version=QUICK_SUITE_SUITE_VERSION,
            adapter_id=_ADAPTER_ID,
            adapter_version=_ADAPTER_VERSION,
        )
        runs.append(
            BenchmarkRunResult(
                run_id=f"run-{task_id}",
                source_id=source_id,
                source_revision="test-revision",
                suite_id=QUICK_SUITE_SUITE_ID,
                suite_version=QUICK_SUITE_SUITE_VERSION,
                adapter_id=_ADAPTER_ID,
                adapter_version=_ADAPTER_VERSION,
                task_attempts=[attempt],
                started_at=datetime.now(UTC),
                finished_at=datetime.now(UTC),
            )
        )
    return runs


def make_benchmark_runs_json(*, failure: int = 0, ungradable: int = 0) -> str:
    """The exact ``{"runs": [...]}`` shape the runner persists."""
    runs = make_benchmark_runs(failure=failure, ungradable=ungradable)
    return json.dumps({"runs": [r.model_dump(mode="json") for r in runs]}, indent=2)


def make_capability_profile(score: float = 0.5, *, coverage_weight: float = _COVERAGE_WEIGHT) -> CapabilityProfile:
    """A profile that qualifies by default: all four enabled dimensions
    measured with ``uncalibrated`` status and the policy's coverage weight."""
    dimensions = tuple(
        DimensionScoreResult(
            dimension=CapabilityDimension(dimension_name),
            status=DimensionScoreStatus.UNCALIBRATED,
            raw_normalized_score=score,
        )
        for _task_id, _source_id, dimension_name in _QUICK_TASKS
    )
    return CapabilityProfile(
        scoring_policy_id=_SCORING_POLICY_ID,
        scoring_policy_version=_SCORING_POLICY_VERSION,
        dimensions=dimensions,
        coverage_weight=coverage_weight,
    )


def commit_run(
    root: Path,
    *,
    execution_id: str = DEFAULT_EXECUTION_ID,
    status: UnifiedRunStatus = UnifiedRunStatus.COMPLETED,
    profile_score: float = 0.5,
    profile: CapabilityProfile | None = None,
    failure: int = 0,
    ungradable: int = 0,
    include_profile: bool = True,
    include_benchmark: bool = True,
    manifest: RunArtifactManifest | None = None,
) -> tuple[RunArtifactRepository, RunArtifactManifest]:
    """Commit a run artifact and return ``(repository, final_manifest)``."""
    repository = RunArtifactRepository(root)
    resolved_manifest = manifest if manifest is not None else make_manifest(execution_id=execution_id, status=status)
    artifacts: dict[str, str] = {}
    if include_profile:
        resolved_profile = profile if profile is not None else make_capability_profile(profile_score)
        artifacts["capability_profile.json"] = resolved_profile.model_dump_json(indent=2)
    if include_benchmark:
        artifacts["benchmark_runs.json"] = make_benchmark_runs_json(failure=failure, ungradable=ungradable)
    final_manifest = repository.commit(resolved_manifest, artifacts)
    return repository, final_manifest


def commit_valid_run(root: Path, **kwargs: object) -> tuple[RunArtifactRepository, RunArtifactManifest]:
    """Convenience alias: a run that passes every qualification gate."""
    return commit_run(root, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ReferenceSnapshot construction
# ---------------------------------------------------------------------------


def make_snapshot(
    *,
    snapshot_id: str,
    model_id: str = "my-real-model",
    provider_id: str = _PROVIDER_ID,
    source_type: str = "operator_verified_api_run",
    suite_sha256: str | None = None,
    execution_id: str | None = DEFAULT_EXECUTION_ID,
    adapter_id: str = _ADAPTER_ID,
    created_at: datetime | None = None,
) -> ReferenceSnapshot:
    """A v0.4-A snapshot with full enhanced provenance (ReferenceSet-ready)."""
    snapshot_created_at = created_at or datetime(2026, 8, 15, tzinfo=UTC)
    if suite_sha256 is None:
        suite_sha256 = get_quick_suite_content_sha256()
    provenance = ReferenceProvenance(
        source_type=source_type,
        created_by="operator",
        created_at=snapshot_created_at,
        suite_sha256=suite_sha256,
        benchmark_revision=f"{QUICK_SUITE_SUITE_ID}-{QUICK_SUITE_SUITE_VERSION}",
        runner_version="0.4.0a0",
        execution_id=execution_id,
        endpoint_redacted="http://test.example.com/v1",
        adapter_id=adapter_id,
        adapter_version=_ADAPTER_VERSION,
        generation_config_sha256=expected_generation_config_sha256(),
        run_manifest_sha256="b" * 64,
        capability_profile_sha256="c" * 64,
        qualification_policy_id=_QUALIFICATION_POLICY_ID,
        qualification_policy_version=_QUALIFICATION_POLICY_VERSION,
        benchmark_revisions={
            "arc_challenge_quick_v1": "ARC-Challenge-2018",
            "humaneval_quick_v1": "human-eval-v1-2021",
            "gsm8k_quick_v1": "gsm8k-main-2023",
            "ifeval_quick_v1": "ifeval-v1-2023",
        },
    )
    return ReferenceSnapshot(
        snapshot_id=snapshot_id,
        model_id=model_id,
        provider_id=provider_id,
        created_at=snapshot_created_at,
        suite_id=QUICK_SUITE_SUITE_ID,
        suite_version=QUICK_SUITE_SUITE_VERSION,
        capability_profile=make_capability_profile(),
        provenance=provenance,
    )
