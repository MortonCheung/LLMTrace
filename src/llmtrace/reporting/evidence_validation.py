"""Evidence reference validation for benchmark reports.

Validates that benchmark task evidence_refs are resolvable against
the top-level evidence array in a JSON report.
"""

from __future__ import annotations

from collections.abc import Sequence

from llmtrace.models.evidence import HTTPEvidence
from llmtrace.reporting.benchmark_models import BenchmarkReportSection


def validate_report_evidence_refs(
    benchmark_sections: Sequence[BenchmarkReportSection],
    evidence: Sequence[HTTPEvidence],
) -> None:
    """Validate evidence reference integrity across the full report.

    Args:
        benchmark_sections: Benchmark report sections (may be empty).
        evidence: Top-level evidence array from AuditResult.

    Raises:
        ValueError: If any benchmark task references an evidence_id
            not present in *evidence*, or if duplicate evidence_id
            values are detected.
    """
    # -- Duplicate check first (so we can report it cleanly) --
    evidence_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for ev in evidence:
        eid = str(ev.evidence_id)
        if eid in evidence_ids:
            duplicate_ids.add(eid)
        evidence_ids.add(eid)

    if duplicate_ids:
        raise ValueError(f"Duplicate evidence_id(s) found in top-level evidence: {sorted(duplicate_ids)}")

    # -- Resolve every benchmark task evidence_ref ---
    for section in benchmark_sections:
        for task in section.tasks:
            for ref in task.evidence_refs:
                if ref not in evidence_ids:
                    raise ValueError(
                        f"Benchmark task '{task.task_id}' references "
                        f"unresolvable evidence_id '{ref}' — "
                        f"not present in top-level evidence"
                    )

    # -- No orphan evidence_refs check (the loop above covers it);
    #    extra evidence_ids with no referrers are allowed (they may
    #    belong to audit findings or be informational).
