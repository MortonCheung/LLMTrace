"""Fingerprint reference aggregation（Task 15）.

一个 model identity 可以有多次 capture（多个 snapshot）。第一版明确不用"只看最新
一次"，也不用"按总样本数加权"——那会让一次超大 capture 支配全部历史。规则是：

    同 identity 的每个 snapshot 的 probabilities → arithmetic mean

即**每次 capture 等权**。

identity 定义为 ``(provider_id, model_id)``，与能力域的
``aggregate_reference_identities`` 保持一致（同域内不同 provider 的同名模型不合并）。

覆盖纪律（fail closed with coverage）：

- 参与聚合的 snapshot 必须自校验内容身份，且与传入套件的 suite id / version /
  content hash、normalizer policy、generation config、repetitions 完全一致；
- 任一 snapshot 缺少套件里的 probe、或某 probe 无有效样本，都直接失败，
  不静默缩小分母。

本模块只产出"聚合后的行为分布"，不产出任何 identity 结论（Rule 3）。
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from llmtrace.fingerprint.models import (
    AggregatedProbeDistribution,
    FingerprintSuite,
    ProbeDistribution,
    normalize_sha256,
)
from llmtrace.fingerprint.reference import FingerprintReferenceSnapshot
from llmtrace.fingerprint.suite import compute_generation_config_sha256, verify_fingerprint_suite


class FingerprintAggregationError(Exception):
    """多次 capture 无法被诚实地聚合成一个参考分布."""

    error_code = "FINGERPRINT_AGGREGATION_ERROR"


def mean_probability(values: Sequence[float]) -> float:
    """每次 capture 等权的算术平均（Task 15）.

    Raises:
        FingerprintAggregationError: 没有可平均的取值（空输入不允许被当成 0）。
    """
    if not values:
        raise FingerprintAggregationError("mean_probability requires at least one capture probability")
    return sum(values) / len(values)


class AggregatedFingerprintReference(BaseModel):
    """同一 identity 多次 capture 聚合出的参考（Task 15）.

    这是匹配（Task 16）与验证（Task 18）使用的参考侧输入；它不声明任何 identity
    结论，只声明"这些 probe 上，该 identity 的历史行为分布是什么"。
    """

    provider_id: str = Field(..., min_length=1)
    model_id: str = Field(..., min_length=1, description="model label; NOT an identity verdict")

    suite_id: str = Field(..., min_length=1)
    suite_version: str = Field(..., min_length=1)
    suite_content_sha256: str

    normalization_policy_id: str = Field(..., min_length=1)
    normalization_policy_version: str = Field(..., min_length=1)

    generation_config_sha256: str

    repetitions: int = Field(..., ge=1)

    snapshot_ids: tuple[str, ...]
    snapshot_count: int = Field(..., ge=1)

    distributions: tuple[AggregatedProbeDistribution, ...]

    model_config = {"frozen": True, "extra": "forbid"}

    @field_validator("suite_content_sha256", "generation_config_sha256")
    @classmethod
    def _validate_sha256(cls, v: str, info: ValidationInfo) -> str:
        field_name = info.field_name
        assert field_name is not None
        return normalize_sha256(v, field_name)

    @model_validator(mode="after")
    def _validate_coverage(self) -> AggregatedFingerprintReference:
        if not self.distributions:
            raise ValueError("an aggregated fingerprint reference must carry at least one probe distribution")
        probe_ids = [distribution.probe_id for distribution in self.distributions]
        if len(set(probe_ids)) != len(probe_ids):
            raise ValueError(f"duplicate probe_id in aggregated distributions: {probe_ids}")

        if len(set(self.snapshot_ids)) != len(self.snapshot_ids):
            raise ValueError(f"snapshot_ids must be unique, got {self.snapshot_ids!r}")
        if len(self.snapshot_ids) != self.snapshot_count:
            raise ValueError(
                f"snapshot_count ({self.snapshot_count}) must equal the number of contributing captures "
                f"({len(self.snapshot_ids)})"
            )
        return self


def aggregate_fingerprint_reference(
    *,
    snapshots: Sequence[FingerprintReferenceSnapshot],
    suite: FingerprintSuite,
) -> AggregatedFingerprintReference:
    """把同一 identity 的多次 capture 聚合成一个参考分布.

    聚合顺序按 ``snapshot_id`` 排序，保证相同输入得到逐字节相同的结果（浮点加法
    不满足结合律，顺序必须确定）。

    Raises:
        FingerprintAggregationError: 输入为空、混合了多个 identity、与套件不兼容、
            重复轮数不一致，或 probe 覆盖不完整。
        FingerprintReferenceError: 某个 snapshot 的内容身份自校验失败。
        FingerprintSuiteIntegrityError: 套件内容身份自校验失败。
    """
    verify_fingerprint_suite(suite)

    if not snapshots:
        raise FingerprintAggregationError("cannot aggregate a fingerprint reference from zero captures")

    ordered = sorted(snapshots, key=lambda snapshot: snapshot.snapshot_id)
    identities = {(snapshot.provider_id, snapshot.model_id) for snapshot in ordered}
    if len(identities) != 1:
        raise FingerprintAggregationError(
            f"aggregation must stay within one identity (provider_id, model_id), got {sorted(identities)!r}"
        )

    expected = (
        suite.suite_id,
        suite.suite_version,
        suite.content_sha256,
        suite.normalization_policy_id,
        suite.normalization_policy_version,
        compute_generation_config_sha256(suite),
    )
    for snapshot in ordered:
        snapshot.verify_content_hash()
        declared = (
            snapshot.suite_id,
            snapshot.suite_version,
            snapshot.suite_content_sha256,
            snapshot.normalization_policy_id,
            snapshot.normalization_policy_version,
            snapshot.generation_config_sha256,
        )
        if declared != expected:
            raise FingerprintAggregationError(
                f"fingerprint snapshot '{snapshot.snapshot_id}' is not compatible with suite "
                f"'{suite.suite_id}' v{suite.suite_version}: declared (suite, policy, generation config) "
                f"identity differs from the suite being aggregated"
            )

    repetitions = {snapshot.repetitions for snapshot in ordered}
    if len(repetitions) != 1:
        raise FingerprintAggregationError(
            f"fingerprint snapshots of one identity must share the same repetitions, got {sorted(repetitions)!r}"
        )

    distributions = tuple(_aggregate_probe(probe_id, suite, ordered) for probe_id in suite.probe_ids)
    first = ordered[0]
    return AggregatedFingerprintReference(
        provider_id=first.provider_id,
        model_id=first.model_id,
        suite_id=suite.suite_id,
        suite_version=suite.suite_version,
        suite_content_sha256=suite.content_sha256,
        normalization_policy_id=suite.normalization_policy_id,
        normalization_policy_version=suite.normalization_policy_version,
        generation_config_sha256=compute_generation_config_sha256(suite),
        repetitions=next(iter(repetitions)),
        snapshot_ids=tuple(snapshot.snapshot_id for snapshot in ordered),
        snapshot_count=len(ordered),
        distributions=distributions,
    )


def _aggregate_probe(
    probe_id: str,
    suite: FingerprintSuite,
    snapshots: Sequence[FingerprintReferenceSnapshot],
) -> AggregatedProbeDistribution:
    """聚合单个 probe；缺项 / 空样本 / support 不一致都 fail closed."""
    per_capture: list[ProbeDistribution] = []
    for snapshot in snapshots:
        by_probe_id = {distribution.probe_id: distribution for distribution in snapshot.distributions}
        distribution = by_probe_id.get(probe_id)
        if distribution is None:
            raise FingerprintAggregationError(
                f"fingerprint snapshot '{snapshot.snapshot_id}' carries no distribution for probe "
                f"'{probe_id}' declared by suite '{suite.suite_id}' v{suite.suite_version}"
            )
        if distribution.sample_count == 0:
            raise FingerprintAggregationError(
                f"fingerprint snapshot '{snapshot.snapshot_id}' has no valid samples for probe '{probe_id}'; "
                f"refusing to average an empty capture into the reference"
            )
        per_capture.append(distribution)

    support = per_capture[0].support
    for snapshot, distribution in zip(snapshots, per_capture, strict=True):
        if distribution.support != support:
            raise FingerprintAggregationError(
                f"support mismatch for probe '{probe_id}' in fingerprint snapshot "
                f"'{snapshot.snapshot_id}': {distribution.support!r} != {support!r}"
            )

    probabilities = {
        item: mean_probability([distribution.probabilities[item] for distribution in per_capture]) for item in support
    }
    return AggregatedProbeDistribution(
        probe_id=probe_id,
        support=support,
        probabilities=probabilities,
        snapshot_ids=tuple(snapshot.snapshot_id for snapshot in snapshots),
        snapshot_count=len(snapshots),
        sample_count=sum(distribution.sample_count for distribution in per_capture),
    )


__all__: list[str] = [
    "FingerprintAggregationError",
    "AggregatedFingerprintReference",
    "mean_probability",
    "aggregate_fingerprint_reference",
]
