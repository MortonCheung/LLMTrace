"""Fingerprint domain models — 与 Capability 域完全独立的身份证据模型。

设计边界（Task 3 invariant）：

- 不向 ``ReferenceSet`` / ``ReferenceSnapshot`` / ``CapabilityProfile`` /
  ``CalibrationPolicy`` 塞入任何 fingerprint 字段；
- 本 package 只使用自己的模型，两个域通过 artifact 层并列，而非继承/混用。

语义纪律（Task 48）：模型与报告文本只允许"行为与参考一致/不一致"这类表述，
不允许出现"检测到假模型 / 已证明作弊"这类定罪式措辞。
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

# 归一化失败的保留取值：必须留在 support 中，绝不缩小分母（Task 8.4 / Task 50）。
INVALID_OUTCOME = "__INVALID__"

#: 聚合概率和归一容差：算术平均不会精确得到 1.0，但误差必须只来自浮点。
_PROBABILITY_SUM_TOLERANCE = 1e-6

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def normalize_sha256(value: str, field_name: str) -> str:
    """要求真实 64 位十六进制 SHA-256，并统一为小写."""
    if not _SHA256_RE.match(value):
        raise ValueError(f"{field_name} must be exactly 64 hexadecimal characters (SHA-256), got {value!r}")
    return value.lower()


class FingerprintSourceRole(StrEnum):
    """参考快照的来源角色（决定它能否进入可信指纹集）."""

    TRUSTED_REFERENCE = "trusted_reference"
    OFFICIAL_BASELINE = "official_baseline"
    TEST_FIXTURE = "test_fixture"
    #: 被测端点当次 run 的采集（Task 21/26）。它是**候选**而不是参考：
    #: 永远不得进入可信指纹集，否则参考库会被被测端点自己的行为污染。
    CANDIDATE_CAPTURE = "candidate_capture"


class FingerprintMatchStatus(StrEnum):
    """指纹匹配的结论等级；只有经过 held-out 验证的 policy 才能给出前两者."""

    RANKED_ONLY = "ranked_only"
    CONSISTENT_WITH_CLAIM = "behavior_consistent_with_claim"
    INCONSISTENT_WITH_CLAIM = "behavior_inconsistent_with_claim"
    INCONCLUSIVE = "inconclusive"


class FingerprintProfile(StrEnum):
    """成本档位：只表示请求成本，不表示准确率或置信度（Task 23）."""

    QUICK = "quick"
    STANDARD = "standard"
    RESEARCH = "research"


#: 每个 profile 的重复轮数。名字只表示成本，不表示 accuracy / confidence。
FINGERPRINT_PROFILE_REPETITIONS: dict[FingerprintProfile, int] = {
    FingerprintProfile.QUICK: 4,
    FingerprintProfile.STANDARD: 8,
    FingerprintProfile.RESEARCH: 16,
}


def repetitions_for_profile(profile: FingerprintProfile) -> int:
    """返回 profile 对应的重复轮数."""
    return FINGERPRINT_PROFILE_REPETITIONS[profile]


class FingerprintProbe(BaseModel):
    """单个低熵分类探测项（第一版只做 low-entropy categorical）."""

    probe_id: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)

    choices: tuple[str, ...] = Field(..., min_length=2)

    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=8, ge=1, le=32)

    weight: float = Field(default=1.0, gt=0.0)

    tags: tuple[str, ...] = ()

    model_config = {"frozen": True, "extra": "forbid"}

    @field_validator("choices")
    @classmethod
    def _validate_choices(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        """选项必须非空且互不重复，否则 support 无法作为概率分布的键."""
        if any(not choice for choice in v):
            raise ValueError("probe choices must be non-empty strings")
        if len(set(v)) != len(v):
            raise ValueError(f"probe choices must be unique, got {v!r}")
        if INVALID_OUTCOME in v:
            raise ValueError(f"probe choices must not contain the reserved outcome {INVALID_OUTCOME!r}")
        return v


class FingerprintSuite(BaseModel):
    """版本化的探测套件；content_sha256 是套件内容身份（自校验）."""

    suite_id: str = Field(..., min_length=1)
    suite_version: str = Field(..., min_length=1)

    normalization_policy_id: str = Field(..., min_length=1)
    normalization_policy_version: str = Field(..., min_length=1)

    probes: tuple[FingerprintProbe, ...]

    content_sha256: str

    model_config = {"frozen": True, "extra": "forbid"}

    @field_validator("content_sha256")
    @classmethod
    def _validate_content_sha256(cls, v: str) -> str:
        return normalize_sha256(v, "content_sha256")

    @model_validator(mode="after")
    def _validate_probes(self) -> FingerprintSuite:
        if not self.probes:
            raise ValueError("a fingerprint suite must declare at least one probe")
        probe_ids = [probe.probe_id for probe in self.probes]
        if len(set(probe_ids)) != len(probe_ids):
            raise ValueError(f"duplicate probe_id in suite: {probe_ids}")
        return self

    @property
    def probe_ids(self) -> tuple[str, ...]:
        """按声明顺序的 probe id 元组."""
        return tuple(probe.probe_id for probe in self.probes)

    def probe_by_id(self, probe_id: str) -> FingerprintProbe:
        """按 id 取 probe；缺失时抛 KeyError（调用方负责 fail closed）."""
        for probe in self.probes:
            if probe.probe_id == probe_id:
                return probe
        raise KeyError(probe_id)


class FingerprintSampleObservation(BaseModel):
    """单次探测请求的观测；不保存完整原始输出（原文由 Evidence 系统管理）."""

    probe_id: str

    sequence_index: int = Field(..., ge=0)
    round_index: int = Field(..., ge=0)

    outcome: str
    valid: bool

    response_model: str | None = None
    finish_reason: str | None = None

    total_latency_ms: float | None = Field(default=None, ge=0.0)
    first_token_latency_ms: float | None = Field(default=None, ge=0.0)

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)

    response_body_sha256: str
    evidence_ref: str

    model_config = {"frozen": True, "extra": "forbid"}

    @field_validator("evidence_ref")
    @classmethod
    def _validate_evidence_ref(cls, v: str) -> str:
        """evidence_ref 必须指向本 run 的真实 HTTPEvidence id（Task 27）."""
        if not v:
            raise ValueError("evidence_ref must not be empty")
        return v

    @model_validator(mode="after")
    def _validate_outcome(self) -> FingerprintSampleObservation:
        """valid=False 的样本必须携带 INVALID_OUTCOME，避免出现"无效却记成某选项"."""
        if not self.valid and self.outcome != INVALID_OUTCOME:
            raise ValueError(f"invalid observation must carry outcome {INVALID_OUTCOME!r}, got {self.outcome!r}")
        return self


class ProbeDistribution(BaseModel):
    """单个 probe 的离散输出分布；support 恒为 ``choices + __INVALID__``."""

    probe_id: str

    support: tuple[str, ...]
    counts: dict[str, int]
    probabilities: dict[str, float]

    sample_count: int = Field(..., ge=0)
    invalid_count: int = Field(..., ge=0)

    model_config = {"frozen": True, "extra": "forbid"}

    @model_validator(mode="after")
    def _validate_support(self) -> ProbeDistribution:
        if len(self.support) < 2:
            raise ValueError("distribution support must contain at least one choice plus the invalid outcome")
        if self.support[-1] != INVALID_OUTCOME:
            raise ValueError(f"distribution support must end with the reserved outcome {INVALID_OUTCOME!r}")
        if len(set(self.support)) != len(self.support):
            raise ValueError(f"distribution support must be unique, got {self.support!r}")
        missing = [item for item in self.support if item not in self.counts or item not in self.probabilities]
        if missing:
            raise ValueError(f"counts/probabilities must cover the whole support; missing {missing!r}")
        extra = [item for item in self.counts if item not in self.support]
        if extra:
            raise ValueError(f"counts contain outcomes outside the support: {extra!r}")
        if sum(self.counts.values()) != self.sample_count:
            raise ValueError(f"counts sum ({sum(self.counts.values())}) must equal sample_count ({self.sample_count})")
        if self.counts.get(INVALID_OUTCOME, 0) != self.invalid_count:
            raise ValueError("invalid_count must equal the count of the reserved invalid outcome")
        return self

    def ordered_probabilities(self) -> tuple[float, ...]:
        """按 support 顺序输出概率，供 JSD 使用（禁止依赖 dict 迭代序，Task 10.2）."""
        return tuple(self.probabilities[item] for item in self.support)


class AggregatedProbeDistribution(BaseModel):
    """同一 identity 多次 capture 聚合出的单 probe 分布（Task 15）.

    ``probabilities`` 是各次 capture 概率的**算术平均**：每次 capture 等权，
    不按样本总数加权，避免一次超大 capture 支配全部历史（Task 15）。

    与 ``ProbeDistribution`` 的区别：这里没有整数 ``counts``。均值不是计数，
    本模型不伪造计数，只保留贡献的 capture 列表与原始样本总数，供覆盖率报告
    与"这次比较用了几次采集"这类审计问题使用。
    """

    probe_id: str

    support: tuple[str, ...]
    probabilities: dict[str, float]

    snapshot_ids: tuple[str, ...]
    snapshot_count: int = Field(..., ge=1)

    sample_count: int = Field(..., ge=0)

    model_config = {"frozen": True, "extra": "forbid"}

    @model_validator(mode="after")
    def _validate_aggregated_distribution(self) -> AggregatedProbeDistribution:
        if len(self.support) < 2:
            raise ValueError("distribution support must contain at least one choice plus the invalid outcome")
        if self.support[-1] != INVALID_OUTCOME:
            raise ValueError(f"distribution support must end with the reserved outcome {INVALID_OUTCOME!r}")
        if len(set(self.support)) != len(self.support):
            raise ValueError(f"distribution support must be unique, got {self.support!r}")

        missing = [item for item in self.support if item not in self.probabilities]
        if missing:
            raise ValueError(f"probabilities must cover the whole support; missing {missing!r}")
        extra = [item for item in self.probabilities if item not in self.support]
        if extra:
            raise ValueError(f"probabilities contain outcomes outside the support: {extra!r}")

        for item in self.support:
            value = self.probabilities[item]
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"probability for {item!r} must be in [0, 1], got {value!r}")
        total = sum(self.probabilities[item] for item in self.support)
        if abs(total - 1.0) > _PROBABILITY_SUM_TOLERANCE:
            raise ValueError(f"aggregated probabilities must sum to 1, got {total!r}")

        if len(set(self.snapshot_ids)) != len(self.snapshot_ids):
            raise ValueError(f"snapshot_ids must be unique, got {self.snapshot_ids!r}")
        if len(self.snapshot_ids) != self.snapshot_count:
            raise ValueError(
                f"snapshot_count ({self.snapshot_count}) must equal the number of contributing captures "
                f"({len(self.snapshot_ids)})"
            )
        return self

    def ordered_probabilities(self) -> tuple[float, ...]:
        """按 support 顺序输出概率，供 JSD 使用（与 ``ProbeDistribution`` 同语义）."""
        return tuple(self.probabilities[item] for item in self.support)
