"""Fingerprint matcher（Task 16 / Task 20）.

把一次 candidate capture 与"同一套件下的参考库"逐个 identity 比对，输出**排序**的
行为距离证据，并按 Task 20 的规则决定结论等级。

语义边界（Rule 3 / Rule 7）：

- entry 里的 ``model_id`` 是**参考条目的标签**，含义是"行为与之最接近的参考"，
  不是 "actual upstream model"；
- ``similarity`` 只是 ``1 - distance`` 的可读写法，不是概率、置信度或准确率；
- 只有 ``policy.validated == True`` 且存在 claimed model 的参考时，才允许输出
  ``BEHAVIOR_CONSISTENT_WITH_CLAIM`` / ``BEHAVIOR_INCONSISTENT_WITH_CLAIM``；
  否则只能是 ``RANKED_ONLY``（有排序、无判定）或 ``INCONCLUSIVE``（没有可比参考）。

本模块不硬编码任何阈值（Task 17）：阈值只能来自 Task 18 的 held-out 验证产物。
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field, model_validator

from llmtrace.fingerprint.aggregation import AggregatedFingerprintReference
from llmtrace.fingerprint.distance import (
    BehavioralDistribution,
    FingerprintDistance,
    FingerprintDistanceError,
    ProbeDistance,
    compute_fingerprint_distance,
)
from llmtrace.fingerprint.models import FingerprintMatchStatus, FingerprintSuite
from llmtrace.fingerprint.policy import FingerprintDecisionPolicy
from llmtrace.fingerprint.suite import verify_fingerprint_suite

#: 报告里必须与证据一同出现的免责声明（行为证据 ≠ 密码学证明）。
MATCH_DISCLAIMER = "Behavioral fingerprint evidence only; not cryptographic proof of upstream model identity."

#: ``similarity = 1 - distance`` 的构造容差。
_SIMILARITY_TOLERANCE = 1e-9


class FingerprintMatchError(Exception):
    """匹配无法在诚实前提下完成（输入与 policy / 套件 / 参考库不一致）."""

    error_code = "FINGERPRINT_MATCH_ERROR"


class FingerprintMatchEntry(BaseModel):
    """一个参考 identity 的比中结果；标签只是"行为最接近的参考"（Rule 3）."""

    model_id: str = Field(..., min_length=1)
    provider_id: str = Field(..., min_length=1)

    distance: float = Field(..., ge=0.0, le=1.0)
    similarity: float = Field(..., ge=0.0, le=1.0)

    comparable_probes: int = Field(..., ge=1)

    #: 聚合距离背后的逐 probe JSD（Task 44）；不可比的 probe 以显式 note 出现，
    #: 因此比较的分母（哪些 probe 参与、哪些没有）永远可审计。
    per_probe: tuple[ProbeDistance, ...] = ()

    model_config = {"frozen": True, "extra": "forbid"}

    @model_validator(mode="after")
    def _validate_similarity(self) -> FingerprintMatchEntry:
        expected = 1.0 - self.distance
        if abs(self.similarity - expected) > _SIMILARITY_TOLERANCE:
            raise ValueError(
                f"similarity ({self.similarity}) must equal 1 - distance ({expected}) for probe-ranked evidence"
            )
        return self

    @model_validator(mode="after")
    def _validate_per_probe(self) -> FingerprintMatchEntry:
        if self.per_probe and sum(1 for probe in self.per_probe if probe.comparable) != self.comparable_probes:
            raise ValueError(
                f"comparable_probes ({self.comparable_probes}) must equal the number of comparable per_probe entries"
            )
        return self


class FingerprintMatchResult(BaseModel):
    """排序后的行为匹配结果与结论等级（Task 16 / Task 20）."""

    entries: tuple[FingerprintMatchEntry, ...]

    status: FingerprintMatchStatus

    claimed_model_id: str | None = None
    claimed_reference_distance: float | None = Field(default=None, ge=0.0, le=1.0)

    policy_id: str | None = None
    policy_version: str | None = None

    experimental: bool = True

    disclaimer: str = MATCH_DISCLAIMER

    model_config = {"frozen": True, "extra": "forbid"}

    @model_validator(mode="after")
    def _validate_status_consistency(self) -> FingerprintMatchResult:
        # 排序必须确定：距离升序，同距离按 identity 升序。
        ordered = sorted(self.entries, key=lambda entry: (entry.distance, entry.provider_id, entry.model_id))
        if list(self.entries) != ordered:
            raise ValueError("match entries must be sorted by ascending (distance, provider_id, model_id)")

        distance_keys = [(entry.provider_id, entry.model_id) for entry in self.entries]
        if len(set(distance_keys)) != len(distance_keys):
            raise ValueError(f"match entries must reference distinct identities, got {distance_keys!r}")

        if self.status in (
            FingerprintMatchStatus.CONSISTENT_WITH_CLAIM,
            FingerprintMatchStatus.INCONSISTENT_WITH_CLAIM,
        ):
            if self.policy_id is None or self.policy_version is None:
                raise ValueError("a claim-verdict status must name the validated policy it came from")
            if self.claimed_model_id is None or self.claimed_reference_distance is None:
                raise ValueError("a claim-verdict status requires a claimed model and its reference distance")
        return self


class FingerprintVerificationResult(BaseModel):
    """本次 run 的 Rule 2 门禁记录：这个判定等级由什么支撑（Task 20 / Task 26）.

    ``policy`` 为 None 表示没有可用于判定的 policy；即使 policy 存在但未通过
    LLMTrace 自身的 held-out 验证，``claim_verdict_produced`` 也必须为 False。
    """

    match_status: FingerprintMatchStatus

    claim_verdict_produced: bool

    #: 本次使用的判定规则全量内容（含 held-out 验证统计）；None 表示没有可用 policy。
    policy: FingerprintDecisionPolicy | None = None

    minimum_comparable_probes: int = Field(..., ge=1)
    reference_identity_count: int = Field(..., ge=0)

    #: 为什么（没有）产生 claim verdict；与 run warnings 同源。
    notes: tuple[str, ...] = ()

    disclaimer: str = MATCH_DISCLAIMER

    model_config = {"frozen": True, "extra": "forbid"}

    @model_validator(mode="after")
    def _validate_gate_consistency(self) -> FingerprintVerificationResult:
        claimed = self.match_status in (
            FingerprintMatchStatus.CONSISTENT_WITH_CLAIM,
            FingerprintMatchStatus.INCONSISTENT_WITH_CLAIM,
        )
        if claimed != self.claim_verdict_produced:
            raise ValueError(
                f"claim_verdict_produced ({self.claim_verdict_produced}) contradicts match_status "
                f"({self.match_status.value!r})"
            )
        if claimed and (self.policy is None or not self.policy.validated):
            raise ValueError("a claim verdict requires a validated decision policy (Rule 2)")
        return self


class FingerprintMatcher:
    """把 candidate capture 与参考库比对的匹配器（Task 16）."""

    def __init__(self, *, suite: FingerprintSuite) -> None:
        verify_fingerprint_suite(suite)
        self._suite = suite

    @property
    def suite(self) -> FingerprintSuite:
        return self._suite

    def match(
        self,
        *,
        candidate: Sequence[BehavioralDistribution],
        references: Sequence[AggregatedFingerprintReference],
        minimum_comparable_probes: int,
        claimed_model_id: str | None = None,
        claimed_provider_id: str | None = None,
        policy: FingerprintDecisionPolicy | None = None,
    ) -> FingerprintMatchResult:
        """计算 candidate 与每个参考 identity 的距离并给出结论等级.

        Args:
            candidate: 本次 run 的分布（单次 capture，Task 9）。
            references: 参考库；每个元素是同一 identity 多次 capture 的聚合（Task 15）。
            minimum_comparable_probes: 可比 probe 下限，必须显式给出（Task 11）。
                传入 policy 时必须与 ``policy.minimum_comparable_probes`` 一致。
            claimed_model_id: 被审计端点声称的 model label；缺省则只能给 RANKED_ONLY。
            claimed_provider_id: 参考库里被声称端点的 provider 标签；给出时按
                ``(provider_id, model_id)`` 精确配对，否则只按 model label 配对。
            policy: 判定规则；未提供或未验证时一律不产生 claim verdict（Rule 2）。

        Raises:
            FingerprintMatchError: 输入与 policy / 套件 / 参考库不一致，或某个参考
                的可比 probe 低于下限（fail closed，不做静默跳过）。
        """
        self._assert_inputs(policy, minimum_comparable_probes)
        if claimed_model_id is None and claimed_provider_id is not None:
            raise FingerprintMatchError("claimed_provider_id requires claimed_model_id")

        distances = self._distance_by_identity(candidate, references, minimum_comparable_probes)

        entries = tuple(
            FingerprintMatchEntry(
                model_id=model_id,
                provider_id=provider_id,
                distance=distance.distance,
                similarity=1.0 - distance.distance,
                comparable_probes=distance.comparable_probes,
                per_probe=distance.per_probe,
            )
            for (provider_id, model_id), distance in sorted(
                distances.items(),
                key=lambda item: (item[1].distance, item[0][0], item[0][1]),
            )
        )

        claimed_reference_distance = self._claimed_distance(
            distances, claimed_model_id=claimed_model_id, claimed_provider_id=claimed_provider_id
        )
        status = _resolve_status(
            entries=entries,
            claimed_reference_distance=claimed_reference_distance,
            policy=policy,
        )

        return FingerprintMatchResult(
            entries=entries,
            status=status,
            claimed_model_id=claimed_model_id,
            claimed_reference_distance=claimed_reference_distance,
            policy_id=policy.policy_id if policy is not None else None,
            policy_version=policy.policy_version if policy is not None else None,
        )

    # -- Internals ---------------------------------------------------------

    def _assert_inputs(
        self,
        policy: FingerprintDecisionPolicy | None,
        minimum_comparable_probes: int,
    ) -> None:
        if minimum_comparable_probes < 1:
            raise FingerprintMatchError(f"minimum_comparable_probes must be >= 1, got {minimum_comparable_probes}")
        if policy is None:
            return
        if policy.minimum_comparable_probes != minimum_comparable_probes:
            raise FingerprintMatchError(
                f"minimum_comparable_probes ({minimum_comparable_probes}) contradicts policy "
                f"'{policy.policy_id}' v{policy.policy_version} ({policy.minimum_comparable_probes})"
            )
        if policy.suite_content_sha256 != self._suite.content_sha256:
            raise FingerprintMatchError(
                f"policy '{policy.policy_id}' v{policy.policy_version} was validated against suite hash "
                f"{policy.suite_content_sha256!r}, not the suite being matched ({self._suite.content_sha256!r})"
            )
        policy.verify_content_hash()

    def _assert_reference_compatible(self, reference: AggregatedFingerprintReference, label: str) -> None:
        declared = (reference.suite_id, reference.suite_version, reference.suite_content_sha256)
        expected = (self._suite.suite_id, self._suite.suite_version, self._suite.content_sha256)
        if declared != expected:
            raise FingerprintMatchError(
                f"fingerprint reference {label} declares suite {declared!r}, which is not the suite "
                f"being matched {expected!r}; references from another suite are never ranked"
            )

    def _distance_by_identity(
        self,
        candidate: Sequence[BehavioralDistribution],
        references: Sequence[AggregatedFingerprintReference],
        minimum_comparable_probes: int,
    ) -> dict[tuple[str, str], FingerprintDistance]:
        distances: dict[tuple[str, str], FingerprintDistance] = {}
        for reference in references:
            key = (reference.provider_id, reference.model_id)
            label = f"'{reference.model_id}' from provider '{reference.provider_id}'"
            if key in distances:
                raise FingerprintMatchError(f"duplicate fingerprint reference identity: {key!r}")
            self._assert_reference_compatible(reference, label)
            try:
                distances[key] = compute_fingerprint_distance(
                    probes=self._suite.probes,
                    candidate=candidate,
                    reference=reference.distributions,
                    minimum_comparable_probes=minimum_comparable_probes,
                )
            except FingerprintDistanceError as exc:
                raise FingerprintMatchError(f"fingerprint reference {label} cannot be ranked: {exc}") from exc
        return distances

    @staticmethod
    def _claimed_distance(
        distances: dict[tuple[str, str], FingerprintDistance],
        *,
        claimed_model_id: str | None,
        claimed_provider_id: str | None,
    ) -> float | None:
        """claimed model 的参考距离.

        同一 model label 可能有多条参考（不同 provider 的采集）。取**最小距离**，
        即"与声称标签最接近的那次采集"：这是偏向"行为一致"方向的保守选择，
        不会让判定更容易变成不一致（Rule 3）。
        """
        if claimed_model_id is None:
            return None
        matched = [
            distance.distance
            for (provider_id, model_id), distance in distances.items()
            if model_id == claimed_model_id and (claimed_provider_id is None or provider_id == claimed_provider_id)
        ]
        if not matched:
            return None
        return min(matched)


def _resolve_status(
    *,
    entries: tuple[FingerprintMatchEntry, ...],
    claimed_reference_distance: float | None,
    policy: FingerprintDecisionPolicy | None,
) -> FingerprintMatchStatus:
    """Task 20：没有"已验证 policy + claimed 参考"时只能 RANKED_ONLY / INCONCLUSIVE."""
    if not entries:
        return FingerprintMatchStatus.INCONCLUSIVE
    if policy is None or not policy.validated or policy.distance_threshold is None:
        return FingerprintMatchStatus.RANKED_ONLY
    if claimed_reference_distance is None:
        return FingerprintMatchStatus.RANKED_ONLY
    if claimed_reference_distance <= policy.distance_threshold:
        return FingerprintMatchStatus.CONSISTENT_WITH_CLAIM
    return FingerprintMatchStatus.INCONSISTENT_WITH_CLAIM


__all__: list[str] = [
    "MATCH_DISCLAIMER",
    "FingerprintMatchError",
    "FingerprintMatchEntry",
    "FingerprintMatchResult",
    "FingerprintMatcher",
    "FingerprintVerificationResult",
]
