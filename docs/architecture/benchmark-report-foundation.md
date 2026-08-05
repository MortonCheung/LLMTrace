# Benchmark Report Foundation

## Current scope (v0.2 — first slice)

This module introduces the **data layer** for benchmark run reports. It
maps benchmark execution artefacts (`RunPlan`, `TaskAttempt`, `GradeResult`,
`BenchmarkRunResult`) into JSON-safe report models.

**What is included:**
- `BenchmarkReportSection` — top-level section for one benchmark run
- `TaskReportItem` — per-task row with optional grade
- `FailureReportItem` — structured, safe failure representation
- `BenchmarkRunSummary` — numeric summary counts
- `build_benchmark_report_section()` — mapper from execution artefacts

**What is NOT included (yet):**
- HTML rendering (no template, no CSS)
- CLI integration (no `llmtrace benchmark report` command)
- `total_score` or `capability_score` (aggregation layer not started)
- Dimension-level aggregation (not started)
- LiveBench, EvalPlus, Inspect AI (only lm-eval adapter exists)
- Real API calls (all tests use `FakeProvider`)

## Relationship to existing audit reports

The existing report pipeline (`AuditResult → generate_json_report / generate_html_report`)
is for **model connectivity and behavior audits** (v0.1 Evidence Audit).

Benchmark reports are a **parallel pipeline** for **capability evaluation** (v0.2+).
They share:

- `evidence_refs`: UUID strings referencing `HTTPEvidence` instances
- The same `schema_version = "1.0"`

They differ:

| Aspect | Audit Report | Benchmark Report |
|--------|-------------|-----------------|
| Top-level model | `AuditResult` | `BenchmarkReportSection` |
| Data source | Probes (connectivity, metadata, etc.) | Adapters (lm-eval, future harnesses) |
| Scoring | Risk level, findings | Raw/normalized scores per task |
| Report format | JSON + HTML + Console | JSON only (HTML planned) |

## Smoke tasks

Smoke tasks (e.g., `llmtrace_smoke`) are internal infrastructure tests.
They are excluded from capability scoring (`capability_score_eligible = False`)
and do not appear in formal benchmark conclusions.

## Mapping rules

1. `GradeResult` is joined to `TaskAttempt` by `attempt_id` (1:1)
2. Duplicate `GradeResult` for the same `attempt_id` raises `ValueError`
3. `SUCCESS` attempt without `GradeResult` → marked ungraded, scores `None`
4. `FAILURE` attempt → `raw_score=None`, `normalized_score=None`, failure preserved
5. `UNGRADABLE` `GradeResult` → `grade_status` preserved, scores `None`
6. Evidence UUIDs flow through as strings
7. `actual_requests` from `BenchmarkRunResult.evidence_refs` count
8. `planned/maximum_requests` from `RunPlan.budget`
9. `estimated_cost=None` stays `None` (never forged as 0)
10. No `total_score` or `capability_score` computed

## Next steps

1. JSON report writer (`generate_benchmark_json_report`)
2. Top-level integration into `AuditResult` or a new `BenchmarkReport` container
3. HTML benchmark section
4. Capability score aggregation (weighting, dimensions)
5. Support for additional harnesses (LiveBench, EvalPlus)
