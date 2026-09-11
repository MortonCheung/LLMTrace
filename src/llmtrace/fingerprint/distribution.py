"""Probe 分布构建（Task 9）.

support 恒定且有序：``choices + __INVALID__``。

归一化失败的样本记入 ``__INVALID__`` 而不是被丢弃 —— 分母必须等于真实发出的
请求数（Task 8.4 / Task 50）。这正是"没有缩小 denominator"的可验证形式：
``sample_count`` 恒等于该 probe 的观测条数。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from llmtrace.fingerprint.models import (
    INVALID_OUTCOME,
    FingerprintProbe,
    FingerprintSampleObservation,
    ProbeDistribution,
)


def build_probe_distribution(
    probe: FingerprintProbe,
    observations: Sequence[FingerprintSampleObservation],
) -> ProbeDistribution:
    """把同一 probe 的观测聚合成离散分布（support = choices + ``__INVALID__``）."""
    relevant = [observation for observation in observations if observation.probe_id == probe.probe_id]

    support = (*probe.choices, INVALID_OUTCOME)
    counts = Counter(observation.outcome for observation in relevant)
    total = len(relevant)

    return ProbeDistribution(
        probe_id=probe.probe_id,
        support=support,
        counts={item: counts.get(item, 0) for item in support},
        probabilities={item: (counts.get(item, 0) / total if total else 0.0) for item in support},
        sample_count=total,
        invalid_count=counts.get(INVALID_OUTCOME, 0),
    )


__all__: list[str] = ["build_probe_distribution"]
