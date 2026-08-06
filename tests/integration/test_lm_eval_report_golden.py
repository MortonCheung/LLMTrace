"""Full smoke report golden test — runs in lm-eval integration CI job.

Validates the complete chain:
  FakeProvider → LmEvalAdapter.list_tasks() → build_plan()
  → LmEvalAdapter.run_task() → LmEvalAdapter.normalize_result()
  → BenchmarkRunResult → build_benchmark_report_section()
  → generate_json_report(schema 1.1)

Full JSON comparison against pre-existing golden fixture.
Includes evidence closure assertions and UUID monkeypatch isolation.
Collected by CI glob: tests/integration/test_lm_eval*.py
"""

from __future__ import annotations

import asyncio
import json
import uuid as _uuid_module
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("lm_eval")

from llmtrace.adapters import lm_eval as _lm_eval
from llmtrace.adapters.lm_eval import LmEvalAdapter
from llmtrace.benchmarks.models import BenchmarkRunResult
from llmtrace.benchmarks.planner import build_plan
from llmtrace.config import AuditConfig, AuthStyle, Protocol
from llmtrace.models.audit import AuditResult, RiskLevel
from llmtrace.reporting.benchmark_mapper import build_benchmark_report_section
from llmtrace.reporting.evidence_validation import validate_report_evidence_refs
from llmtrace.reporting.json_report import generate_json_report
from tests.adapters import conftest as _conftest
from tests.adapters.conftest import FakeProvider

# ---------------------------------------------------------------------------
# Fixed UUID sequence (matches gen_smoke_golden.py)
# ---------------------------------------------------------------------------

_uuid_counter = 0


def _make_fixed_uuid4() -> _uuid_module.UUID:
    """Deterministic UUID generator: 00000000-0000-0000-0000-{N:012d}."""
    global _uuid_counter
    _uuid_counter += 1
    return _uuid_module.UUID(f"00000000-0000-0000-0000-{_uuid_counter:012d}")


@pytest.fixture(scope="module")
def golden_fixture_path() -> Path:
    return Path(__file__).parents[1] / "reporting" / "fixtures" / "lm_eval_smoke_full_report_golden.json"


class TestSmokeFullReportGolden:
    """Golden test: full generate_json_report output must match fixture."""

    def test_golden_full_report_matches(
        self, tmp_path: Path, golden_fixture_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert golden_fixture_path.exists(), f"Golden fixture not found at {golden_fixture_path}. It must be committed."

        # Reset UUID counter
        global _uuid_counter
        _uuid_counter = 0

        # Monkeypatch via pytest — cleanly restored after test
        monkeypatch.setattr(_lm_eval, "uuid4", _make_fixed_uuid4)  # type: ignore[arg-type]
        monkeypatch.setattr(_conftest, "uuid4", _make_fixed_uuid4)  # type: ignore[arg-type]

        # Fix time.monotonic for deterministic latency in FakeProvider.complete()

        _fake_mono = [0.0]

        def _mono() -> float:
            _fake_mono[0] += 0.1
            return _fake_mono[0]

        monkeypatch.setattr(_conftest.time, "monotonic", _mono)  # type: ignore[attr-defined]

        async def _run() -> None:
            adapter = LmEvalAdapter()
            task_specs = adapter.list_tasks()
            smoke_spec = task_specs[0]

            plan = build_plan(
                suite_id="llmtrace_smoke",
                suite_version="1.0.0",
                source_id="lm-eval",
                source_revision="0000000-smoke",
                adapter_id=adapter.adapter_id,
                adapter_version=adapter.adapter_version,
                tasks=[smoke_spec],
            )

            provider = FakeProvider(
                response_map={
                    "LLMTRACE_OK": "LLMTRACE_OK",
                    "DETERMINISTIC": "DETERMINISTIC",
                    "ADAPTER_WORKS": "ADAPTER_WORKS",
                    "EVIDENCE_TRACED": "EVIDENCE_TRACED",
                }
            )

            attempt = await adapter.run_task(smoke_spec, provider)

            # -- Evidence closure assertions ---
            assert len(provider.evidence) > 0, "provider must store real evidence"
            assert len(provider.evidence) == len(attempt.evidence_refs), (
                "provider evidence count must match attempt evidence_refs count"
            )
            provider_eids = {str(ev.evidence_id) for ev in provider.evidence}
            attempt_refs = set(attempt.evidence_refs)
            assert provider_eids == attempt_refs, "provider evidence_ids must one-to-one match attempt.evidence_refs"

            metric_result = attempt.metadata["metric_result"]
            assert isinstance(metric_result, dict)

            raw_result: dict[str, object] = {
                "results": {
                    metric_result["task_name"]: {
                        metric_result["metric_name"]: metric_result["value"],
                    },
                },
                "evidence_ids": attempt.evidence_refs,
                "task_name": metric_result["task_name"],
                "attempt_id": attempt.attempt_id,
            }

            grade = adapter.normalize_result(raw_result)

            # Grade -> attempt consistency
            assert grade.evidence_refs == attempt.evidence_refs

            run_result = BenchmarkRunResult(
                run_id=str(_make_fixed_uuid4()),
                task_attempts=[attempt],
                grade_results=[grade],
                evidence_refs=attempt.evidence_refs,
                source_id="lm-eval",
                source_revision="0000000-smoke",
                suite_id="llmtrace_smoke",
                suite_version="1.0.0",
                adapter_id=adapter.adapter_id,
                adapter_version=adapter.adapter_version,
            )

            section = build_benchmark_report_section(plan, run_result)

            # Section -> attempt consistency
            assert section.tasks[0].evidence_refs == attempt.evidence_refs

            audit_result = AuditResult(
                config=AuditConfig(
                    protocol=Protocol.OPENAI,
                    base_url="https://api.example.com",
                    model="test-model",
                    api_key_env="TEST_KEY",
                    auth_style=AuthStyle.BEARER,
                    repeat_count=1,
                    timeout=30.0,
                    max_output_tokens=100,
                    check_streaming=False,
                ),
                evidence=list(provider.evidence),
                findings=[],
                risk_level=RiskLevel.INCONCLUSIVE,
                schema_fingerprints=[],
                model_list=[],
                start_time=datetime(2026, 1, 1, tzinfo=UTC),
                end_time=datetime(2026, 1, 1, 1, tzinfo=UTC),
                llmtrace_version="0.2.0",
                python_version="3.12",
                platform="darwin",
                report_id="golden-report-id",
                content_hash="",
            )

            output_path = tmp_path / "report.json"
            generate_json_report(audit_result, output_path, benchmark_sections=[section])

            actual = json.loads(output_path.read_text())
            expected = json.loads(golden_fixture_path.read_text())

            # -- Top-level evidence <-> benchmark evidence_refs closure ---
            top_level_eids = {item["evidence_id"] for item in actual["evidence"]}
            benchmark_refs = set(actual["benchmarks"][0]["tasks"][0]["evidence_refs"])
            assert benchmark_refs, "benchmark task must reference evidence"
            assert benchmark_refs <= top_level_eids, (
                f"benchmark refs not found in top-level evidence:\n  missing: {benchmark_refs - top_level_eids}"
            )

            # Validation function check
            validate_report_evidence_refs([section], list(provider.evidence))

            assert actual == expected, (
                f"Golden fixture mismatch.\n"
                f"Actual keys: {sorted(actual.keys())}\n"
                f"Expected keys: {sorted(expected.keys())}\n"
                f"Diff in content_hash may indicate evidence content change."
            )

        asyncio.run(_run())


class TestUuidIsolation:
    """Verify that monkeypatch does not leak between tests."""

    def test_original_uuid4_restored_after_patched_test(self) -> None:
        """After the golden test runs with monkeypatch, uuid4 is restored."""
        import uuid as _uuid

        # The original import references must still point to the real uuid4
        assert _lm_eval.uuid4 is _uuid.uuid4, "_lm_eval.uuid4 should be real uuid.uuid4 after monkeypatch cleanup"
        assert _conftest.uuid4 is _uuid.uuid4, "_conftest.uuid4 should be real uuid.uuid4 after monkeypatch cleanup"

    def test_non_deterministic_uuid_in_new_test(self) -> None:
        """A separate test without monkeypatch gets normal UUIDs."""
        provider = FakeProvider(
            response_map={"x": "rx"},
        )
        import asyncio as _asyncio

        async def _call() -> None:
            ev1 = await provider.complete(model="m", messages=[{"role": "user", "content": "x"}])
            ev2 = await provider.complete(model="m", messages=[{"role": "user", "content": "x"}])
            assert str(ev1.evidence_id) != str(ev2.evidence_id), "UUIDs must be unique"
            # Verify they're not from the fixed sequence
            defixed_prefix = "00000000-0000-0000-0000-"
            assert not str(ev1.evidence_id).startswith(defixed_prefix), (
                "UUID should not be from fixed sequence after monkeypatch cleanup"
            )

        _asyncio.run(_call())
