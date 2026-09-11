"""Fingerprint decision policy（Task 19）.

policy 是"LLMTrace 自己验证出来的判定规则"，而不是抄来的数字（Task 17）：

- 只有 ``validated == True`` 的 policy 才允许给出
  ``BEHAVIOR_CONSISTENT_WITH_CLAIM`` / ``BEHAVIOR_INCONSISTENT_WITH_CLAIM``
  （Rule 2 / Task 20）；
- ``validated == False`` 时不得携带 production threshold（Step 18.1）；
- threshold 的合法性由 Step 18.3 的约束保证：``false_accept_rate <= max_far_target``，
  该约束在模型层 fail closed，避免"手工写一个好看的数字"。

与 ``FingerprintReferenceSet`` 一样：immutable / append-only / self-hashed。
"""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from llmtrace.fingerprint.models import normalize_sha256
from llmtrace.utilities.hashing import canonical_json_hash


class FingerprintPolicyError(Exception):
    """Fingerprint policy 错误基类."""

    error_code = "FINGERPRINT_POLICY_ERROR"


class FingerprintPolicyIntegrityError(FingerprintPolicyError):
    """声明的内容身份与重算值不一致."""


class FingerprintDecisionPolicy(BaseModel):
    """指纹判定规则；threshold 只来自 held-out 验证（Rule 2）.

    未验证的 policy 仍然可以携带 top-1 / top-3 accuracy 之类的诊断数字，但不得
    携带 threshold、TPR、FAR —— 那些是"可用于判定"的产物。
    """

    policy_id: str = Field(..., min_length=1)
    policy_version: str = Field(..., min_length=1)

    fingerprint_set_id: str = Field(..., min_length=1)
    fingerprint_set_content_sha256: str

    suite_content_sha256: str

    validated: bool

    distance_threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    #: Task 30：held-out 验证时收集的 ``within_reference_temporal_distances`` 上界。
    #: 只有 validated policy 才允许携带它；未验证时不携带（Rule 2）。验证集里的
    #: reference capture 轮数不足以形成两个时间窗时，它也可以是 None —— 此时
    #: routing 只能把"没有 temporal baseline"记为 limitation。
    temporal_divergence_baseline: float | None = Field(default=None, ge=0.0, le=1.0)

    minimum_comparable_probes: int = Field(..., ge=1)

    max_far_target: float = Field(..., ge=0.0, le=1.0)

    identity_count: int = Field(..., ge=0)
    held_out_capture_count: int = Field(..., ge=0)

    top1_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    top3_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)

    true_positive_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    false_accept_rate: float | None = Field(default=None, ge=0.0, le=1.0)

    content_sha256: str

    model_config = {"frozen": True, "extra": "forbid"}

    @field_validator(
        "fingerprint_set_content_sha256",
        "suite_content_sha256",
        "content_sha256",
    )
    @classmethod
    def _validate_sha256(cls, v: str, info: ValidationInfo) -> str:
        field_name = info.field_name
        assert field_name is not None
        return normalize_sha256(v, field_name)

    @model_validator(mode="after")
    def _validate_validation_state(self) -> FingerprintDecisionPolicy:
        judged = (self.distance_threshold, self.true_positive_rate, self.false_accept_rate)

        if self.validated:
            if any(value is None for value in judged):
                raise ValueError(
                    "a validated fingerprint policy must carry distance_threshold, "
                    "true_positive_rate and false_accept_rate"
                )
            assert self.false_accept_rate is not None
            if self.false_accept_rate > self.max_far_target:
                raise ValueError(
                    f"validated policy violates its own FAR ceiling: false_accept_rate "
                    f"({self.false_accept_rate}) > max_far_target ({self.max_far_target})"
                )
        elif any(value is not None for value in judged):
            raise ValueError(
                "an unvalidated fingerprint policy must not carry distance_threshold, "
                "true_positive_rate or false_accept_rate (no production threshold without validation)"
            )
        if not self.validated and self.temporal_divergence_baseline is not None:
            raise ValueError(
                "an unvalidated fingerprint policy must not carry a temporal divergence baseline "
                "(no reference baseline without validation)"
            )
        return self

    def compute_content_sha256(self) -> str:
        """Canonical self-hash（``content_sha256`` 置空后取 canonical JSON 的 SHA-256）."""
        payload = self.model_dump(mode="json", exclude={"content_sha256"})
        payload["content_sha256"] = ""
        return canonical_json_hash(payload)

    def verify_content_hash(self) -> str:
        """声明值与重算值不一致即 fail closed，返回已验证的摘要."""
        actual = self.compute_content_sha256()
        if actual != self.content_sha256:
            raise FingerprintPolicyIntegrityError(
                f"fingerprint policy '{self.policy_id}' v{self.policy_version} content hash mismatch: "
                f"recomputed {actual!r} != declared {self.content_sha256!r}"
            )
        return actual


def build_fingerprint_policy(
    *,
    policy_id: str,
    policy_version: str,
    fingerprint_set_id: str,
    fingerprint_set_content_sha256: str,
    suite_content_sha256: str,
    validated: bool,
    minimum_comparable_probes: int,
    max_far_target: float,
    identity_count: int,
    held_out_capture_count: int,
    distance_threshold: float | None = None,
    temporal_divergence_baseline: float | None = None,
    top1_accuracy: float | None = None,
    top3_accuracy: float | None = None,
    true_positive_rate: float | None = None,
    false_accept_rate: float | None = None,
) -> FingerprintDecisionPolicy:
    """构造带自哈希的 policy（先以占位摘要走完整校验，再替换自哈希字段）."""
    provisional = FingerprintDecisionPolicy(
        policy_id=policy_id,
        policy_version=policy_version,
        fingerprint_set_id=fingerprint_set_id,
        fingerprint_set_content_sha256=fingerprint_set_content_sha256,
        suite_content_sha256=suite_content_sha256,
        validated=validated,
        distance_threshold=distance_threshold,
        temporal_divergence_baseline=temporal_divergence_baseline,
        minimum_comparable_probes=minimum_comparable_probes,
        max_far_target=max_far_target,
        identity_count=identity_count,
        held_out_capture_count=held_out_capture_count,
        top1_accuracy=top1_accuracy,
        top3_accuracy=top3_accuracy,
        true_positive_rate=true_positive_rate,
        false_accept_rate=false_accept_rate,
        content_sha256="0" * 64,
    )
    policy = provisional.model_copy(update={"content_sha256": provisional.compute_content_sha256()})
    policy.verify_content_hash()
    return policy


__all__: list[str] = [
    "FingerprintPolicyError",
    "FingerprintPolicyIntegrityError",
    "FingerprintDecisionPolicy",
    "build_fingerprint_policy",
]
