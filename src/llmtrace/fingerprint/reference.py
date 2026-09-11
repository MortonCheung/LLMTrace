"""Fingerprint reference snapshot（Task 12）.

一个 snapshot 是"某个 model label 在某个套件版本下的一次行为采集"这一历史事实：

- immutable / append-only / self-hashed；
- 只保存观测与分布，不保存完整原始响应（原文由 Evidence 系统管理，Task 8.5）；
- 与 Capability 域的 ``ReferenceSnapshot`` **完全分离**：两者回答不同问题
  （Rule 1），因此这里不继承、不混用该模型。

内容身份复用仓库既有的 canonical JSON + SHA-256 纪律（``reference_set`` 的 §24
做法），不另造一套 hashing framework。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from llmtrace.fingerprint.distribution import build_probe_distribution
from llmtrace.fingerprint.models import (
    FingerprintSampleObservation,
    FingerprintSourceRole,
    FingerprintSuite,
    ProbeDistribution,
    normalize_sha256,
)
from llmtrace.fingerprint.suite import compute_generation_config_sha256, verify_fingerprint_suite
from llmtrace.utilities.hashing import canonical_json_hash

# snapshot_id 同时是文件名 stem，必须无法越出目录或嵌套。
_SNAPSHOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class FingerprintReferenceError(Exception):
    """Fingerprint 参考数据错误基类."""

    error_code = "FINGERPRINT_REFERENCE_ERROR"


def _require_utc(value: datetime, field_name: str) -> datetime:
    """拒绝 naive datetime，并把 aware datetime 归一到 UTC."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware, got naive datetime {value.isoformat()}")
    return value.astimezone(UTC)


class FingerprintReferenceSnapshot(BaseModel):
    """一次指纹采集的不可变记录（documents one capture of one model label）."""

    snapshot_id: str = Field(..., min_length=1)
    model_id: str = Field(..., min_length=1, description="model label; NOT an identity verdict")
    provider_id: str = Field(..., min_length=1)

    source_role: FingerprintSourceRole

    suite_id: str = Field(..., min_length=1)
    suite_version: str = Field(..., min_length=1)
    suite_content_sha256: str

    normalization_policy_id: str = Field(..., min_length=1)
    normalization_policy_version: str = Field(..., min_length=1)

    generation_config_sha256: str

    repetitions: int = Field(..., ge=1)

    observations: tuple[FingerprintSampleObservation, ...]
    distributions: tuple[ProbeDistribution, ...]

    captured_at: datetime

    content_sha256: str

    model_config = {"frozen": True, "extra": "forbid"}

    @field_validator("snapshot_id")
    @classmethod
    def _validate_snapshot_id(cls, v: str) -> str:
        if not _SNAPSHOT_ID_RE.match(v):
            raise ValueError(f"snapshot_id must match [A-Za-z0-9][A-Za-z0-9._-]* to stay filename-safe, got {v!r}")
        return v

    @field_validator("suite_content_sha256", "generation_config_sha256", "content_sha256")
    @classmethod
    def _validate_sha256(cls, v: str, info: ValidationInfo) -> str:
        field_name = info.field_name
        assert field_name is not None
        return normalize_sha256(v, field_name)

    @field_validator("captured_at")
    @classmethod
    def _validate_captured_at(cls, v: datetime) -> datetime:
        return _require_utc(v, "captured_at")

    @model_validator(mode="after")
    def _validate_distributions(self) -> FingerprintReferenceSnapshot:
        if not self.distributions:
            raise ValueError("a fingerprint snapshot must carry at least one probe distribution")
        probe_ids = [distribution.probe_id for distribution in self.distributions]
        if len(set(probe_ids)) != len(probe_ids):
            raise ValueError(f"duplicate probe_id in snapshot distributions: {probe_ids}")
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
            raise FingerprintReferenceError(
                f"fingerprint snapshot '{self.snapshot_id}' content hash mismatch: "
                f"recomputed {actual!r} != declared {self.content_sha256!r}"
            )
        return actual


def build_fingerprint_snapshot(
    *,
    snapshot_id: str,
    model_id: str,
    provider_id: str,
    source_role: FingerprintSourceRole,
    suite: FingerprintSuite,
    repetitions: int,
    observations: Sequence[FingerprintSampleObservation],
    captured_at: datetime | None = None,
) -> FingerprintReferenceSnapshot:
    """由一次采集的观测构建自哈希 snapshot.

    ``distributions`` 由套件声明顺序的 probe 逐一聚合（support 固定为
    ``choices + __INVALID__``），因此不可比性只能来自套件差异，而不是缺项。

    Raises:
        FingerprintSuiteIntegrityError: 套件内容身份自校验失败。
        FingerprintReferenceError: 观测引用了套件之外的 probe_id。
    """
    verify_fingerprint_suite(suite)

    known = set(suite.probe_ids)
    unknown = sorted({observation.probe_id for observation in observations} - known)
    if unknown:
        raise FingerprintReferenceError(
            f"fingerprint snapshot '{snapshot_id}' carries observations for probes outside suite "
            f"'{suite.suite_id}' v{suite.suite_version}: {unknown}"
        )

    distributions = tuple(build_probe_distribution(probe, observations) for probe in suite.probes)

    # 先用占位 content_sha256 走完整校验，再只替换自哈希字段。
    provisional = FingerprintReferenceSnapshot(
        snapshot_id=snapshot_id,
        model_id=model_id,
        provider_id=provider_id,
        source_role=source_role,
        suite_id=suite.suite_id,
        suite_version=suite.suite_version,
        suite_content_sha256=suite.content_sha256,
        normalization_policy_id=suite.normalization_policy_id,
        normalization_policy_version=suite.normalization_policy_version,
        generation_config_sha256=compute_generation_config_sha256(suite),
        repetitions=repetitions,
        observations=tuple(observations),
        distributions=distributions,
        captured_at=captured_at if captured_at is not None else datetime.now(UTC),
        content_sha256="0" * 64,
    )

    snapshot = provisional.model_copy(update={"content_sha256": provisional.compute_content_sha256()})
    snapshot.verify_content_hash()
    return snapshot


__all__: list[str] = [
    "FingerprintReferenceError",
    "FingerprintReferenceSnapshot",
    "build_fingerprint_snapshot",
]
