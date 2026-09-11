"""Performance summary（Task 34）.

只消费既有证据字段（Rule 5 / Task 34）：

    ``first_token_latency_ms``  → TTFT
    ``total_latency_ms``        → E2E
    ``output_tokens``           → TPOT 的分母

不新建 telemetry framework、不额外发请求、不做插值：某个指标算不出来时它是
``None`` 而不是 0（Step 34.2「缺失不填 0」），并用 coverage 说明"有多少请求
真的贡献了这个数字"。
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean

from pydantic import BaseModel, Field

from llmtrace.models.evidence import HTTPEvidence


def compute_tpot_ms(
    first_token_latency_ms: float | None,
    total_latency_ms: float | None,
    output_tokens: int | None,
) -> float | None:
    """Time per output token：decode 阶段平均每个输出 token 的耗时（Step 34.1）.

    三个输入任一缺失、或 ``output_tokens <= 1``（没有 decode 阶段）时返回 ``None``。
    首 token 晚于整体完成（流式计时误差）时 decode 时间按 0 计，不产生负数。
    """
    if first_token_latency_ms is None:
        return None

    if total_latency_ms is None:
        return None

    if output_tokens is None or output_tokens <= 1:
        return None

    decode_time = total_latency_ms - first_token_latency_ms

    return max(0.0, decode_time) / (output_tokens - 1)


class PerformanceSummary(BaseModel):
    """一次 run 的性能概览；算不出来的指标是 ``None``，不是 0（Step 34.2）."""

    request_count: int = Field(..., ge=0, description="Number of evidence items considered")

    e2e_latency_ms: float | None = Field(default=None, ge=0.0, description="Mean end-to-end (total) latency")
    ttft_ms: float | None = Field(default=None, ge=0.0, description="Mean first-token latency")
    tpot_ms: float | None = Field(default=None, ge=0.0, description="Mean decode latency per output token")

    e2e_coverage: float = Field(..., ge=0.0, le=1.0, description="Share of requests reporting a total latency")
    ttft_coverage: float = Field(..., ge=0.0, le=1.0, description="Share of requests reporting a first-token latency")
    tpot_coverage: float = Field(..., ge=0.0, le=1.0, description="Share of requests where TPOT was computable")

    model_config = {"frozen": True, "extra": "forbid"}


def summarize_performance(evidences: Sequence[HTTPEvidence]) -> PerformanceSummary:
    """按算术平均汇总证据里的性能字段（Step 34.2）.

    coverage 的分母是**传入的全部证据条数**（不因为字段缺失而缩小分母）；
    空输入时三个均值都是 ``None``、coverage 都是 0.0，不伪造 0 毫秒。
    """
    total = len(evidences)

    e2e_values = [item.total_latency_ms for item in evidences if item.total_latency_ms is not None]
    ttft_values = [item.first_token_latency_ms for item in evidences if item.first_token_latency_ms is not None]
    tpot_values = [
        value
        for value in (
            compute_tpot_ms(item.first_token_latency_ms, item.total_latency_ms, item.output_tokens)
            for item in evidences
        )
        if value is not None
    ]

    return PerformanceSummary(
        request_count=total,
        e2e_latency_ms=fmean(e2e_values) if e2e_values else None,
        ttft_ms=fmean(ttft_values) if ttft_values else None,
        tpot_ms=fmean(tpot_values) if tpot_values else None,
        e2e_coverage=len(e2e_values) / total if total else 0.0,
        ttft_coverage=len(ttft_values) / total if total else 0.0,
        tpot_coverage=len(tpot_values) / total if total else 0.0,
    )


__all__: list[str] = [
    "compute_tpot_ms",
    "PerformanceSummary",
    "summarize_performance",
]
