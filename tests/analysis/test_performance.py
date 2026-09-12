"""Performance summary（Task 34）.

Pins the "missing is None, never 0" discipline: a latency metric that cannot be
computed must be ``None`` with a coverage that says so, and the coverage
denominator stays the number of evidence items passed in.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llmtrace.analysis.performance import (
    PerformanceSummary,
    compute_tpot_ms,
    summarize_performance,
)
from llmtrace.models.evidence import HTTPEvidence

from .conftest import make_evidence

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evidence(
    *,
    ttft_ms: float | None = 20.0,
    total_ms: float | None = 100.0,
    output_tokens: int | None = 5,
) -> HTTPEvidence:
    """An HTTPEvidence with controllable latency / token fields."""
    return make_evidence().model_copy(
        update={
            "first_token_latency_ms": ttft_ms,
            "total_latency_ms": total_ms,
            "output_tokens": output_tokens,
        }
    )


# ---------------------------------------------------------------------------
# compute_tpot_ms
# ---------------------------------------------------------------------------


class TestComputeTpotMs:
    def test_decode_time_is_amortized_over_remaining_tokens(self) -> None:
        # (400 - 100) / (4 - 1) == 100.0
        assert compute_tpot_ms(100.0, 400.0, 4) == pytest.approx(100.0)

    def test_single_output_token_has_no_decode_phase(self) -> None:
        assert compute_tpot_ms(100.0, 400.0, 1) is None

    @pytest.mark.parametrize("output_tokens", [0, -3])
    def test_non_positive_output_tokens_are_not_computable(self, output_tokens: int) -> None:
        assert compute_tpot_ms(100.0, 400.0, output_tokens) is None

    def test_inverted_timings_clamp_to_zero_without_going_negative(self) -> None:
        assert compute_tpot_ms(400.0, 100.0, 4) == pytest.approx(0.0)

    def test_equal_timings_mean_no_decode_time(self) -> None:
        assert compute_tpot_ms(100.0, 100.0, 4) == pytest.approx(0.0)

    @pytest.mark.parametrize(
        ("ttft", "total", "tokens"),
        [
            (None, 400.0, 4),
            (100.0, None, 4),
            (100.0, 400.0, None),
            (None, None, None),
        ],
    )
    def test_missing_input_yields_none(self, ttft: float | None, total: float | None, tokens: int | None) -> None:
        assert compute_tpot_ms(ttft, total, tokens) is None


# ---------------------------------------------------------------------------
# summarize_performance
# ---------------------------------------------------------------------------


class TestSummarizePerformance:
    def test_empty_input_reports_none_means_and_zero_coverage(self) -> None:
        summary = summarize_performance([])

        assert summary.request_count == 0
        assert summary.e2e_latency_ms is None
        assert summary.ttft_ms is None
        assert summary.tpot_ms is None
        assert summary.e2e_coverage == 0.0
        assert summary.ttft_coverage == 0.0
        assert summary.tpot_coverage == 0.0

    def test_full_evidence_yields_arithmetic_means(self) -> None:
        summary = summarize_performance([_evidence(), _evidence(ttft_ms=40.0, total_ms=300.0)])

        assert summary.request_count == 2
        # e2e: (100 + 300) / 2 ; ttft: (20 + 40) / 2
        assert summary.e2e_latency_ms == pytest.approx(200.0)
        assert summary.ttft_ms == pytest.approx(30.0)
        # tpot: 20.0 and (300 - 40) / 4 == 65.0 -> mean 42.5
        assert summary.tpot_ms == pytest.approx(42.5)
        assert summary.e2e_coverage == 1.0
        assert summary.ttft_coverage == 1.0
        assert summary.tpot_coverage == 1.0

    def test_coverage_denominator_is_every_evidence_passed_in(self) -> None:
        summary = summarize_performance(
            [
                _evidence(),
                _evidence(ttft_ms=None),
                _evidence(ttft_ms=None, total_ms=None, output_tokens=5),
                _evidence(ttft_ms=None, total_ms=None, output_tokens=None),
            ]
        )

        assert summary.request_count == 4
        # e2e reported by 2 of 4, ttft by 1 of 4, tpot computable for 1 of 4
        assert summary.e2e_coverage == pytest.approx(0.5)
        assert summary.ttft_coverage == pytest.approx(0.25)
        assert summary.tpot_coverage == pytest.approx(0.25)

    def test_no_latency_anywhere_keeps_means_none(self) -> None:
        summary = summarize_performance(
            [_evidence(ttft_ms=None, total_ms=None), _evidence(ttft_ms=None, total_ms=None)]
        )

        assert summary.request_count == 2
        assert summary.e2e_latency_ms is None
        assert summary.ttft_ms is None
        assert summary.tpot_ms is None
        assert summary.e2e_coverage == 0.0
        assert summary.ttft_coverage == 0.0
        assert summary.tpot_coverage == 0.0

    def test_inverted_timing_counts_as_computable_zero(self) -> None:
        summary = summarize_performance([_evidence(ttft_ms=400.0, total_ms=100.0, output_tokens=4)])

        assert summary.tpot_ms == pytest.approx(0.0)
        assert summary.tpot_coverage == 1.0


# ---------------------------------------------------------------------------
# PerformanceSummary model
# ---------------------------------------------------------------------------


class TestPerformanceSummaryModel:
    def test_summary_is_frozen(self) -> None:
        summary = summarize_performance([_evidence()])

        with pytest.raises(ValidationError):
            summary.e2e_latency_ms = 1.0  # type: ignore[misc]

    def test_summary_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            PerformanceSummary(
                request_count=1,
                e2e_coverage=1.0,
                ttft_coverage=1.0,
                tpot_coverage=1.0,
                unexpected="nope",  # type: ignore[call-arg]
            )
