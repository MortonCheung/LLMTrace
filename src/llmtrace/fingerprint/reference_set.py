"""Fingerprint reference set（Task 14）.

独立于 Capability 域的 ``ReferenceSet``：它只回答"哪些行为采集可以互相比较"，
不参与能力评分（Rule 1）。

可信边界（Step 14.2）：``TEST_FIXTURE`` 快照在 production 路径上永远不能进入
trusted fingerprint set。测试需要的例外只通过构造参数开放，**不暴露 CLI flag**。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from llmtrace.fingerprint.models import FingerprintSourceRole, normalize_sha256
from llmtrace.fingerprint.reference import FingerprintReferenceSnapshot
from llmtrace.utilities.hashing import canonical_json_hash


class FingerprintReferenceSetError(Exception):
    """Fingerprint reference set 错误基类."""

    error_code = "FINGERPRINT_REFERENCE_SET_ERROR"


class DuplicateFingerprintReferenceSetError(FingerprintReferenceSetError):
    """同 ``(id, version)`` 已存在；集合不可变且 append-only."""


class FingerprintReferenceSetNotFoundError(FingerprintReferenceSetError):
    """请求的 ``(id, version)`` 不存在."""


class FingerprintReferenceSetIntegrityError(FingerprintReferenceSetError):
    """声明的内容身份与重算值不一致."""


class FingerprintReferenceSetCompatibilityError(FingerprintReferenceSetError):
    """成员快照未通过兼容门禁（套件 / 归一化策略 / 生成配置 / 重复轮数不一致）."""


class FixtureFingerprintReferenceError(FingerprintReferenceSetError):
    """``test_fixture`` 快照被用于 trusted fingerprint set."""


class FingerprintReferenceSetMember(BaseModel):
    """指向一个已持久化 fingerprint snapshot 的不可变指针."""

    snapshot_id: str = Field(..., min_length=1)
    snapshot_sha256: str = Field(..., description="SHA-256 of the persisted snapshot file bytes")

    model_id: str = Field(..., min_length=1)
    provider_id: str = Field(..., min_length=1)

    source_role: FingerprintSourceRole

    model_config = {"frozen": True, "extra": "forbid"}

    @field_validator("snapshot_sha256")
    @classmethod
    def _validate_snapshot_sha256(cls, v: str) -> str:
        return normalize_sha256(v, "snapshot_sha256")


class FingerprintReferenceSet(BaseModel):
    """一组可比 fingerprint snapshot 的版本化集合."""

    fingerprint_set_id: str = Field(..., min_length=1)
    fingerprint_set_version: str = Field(..., min_length=1)

    suite_id: str = Field(..., min_length=1)
    suite_version: str = Field(..., min_length=1)
    suite_content_sha256: str

    normalization_policy_id: str = Field(..., min_length=1)
    normalization_policy_version: str = Field(..., min_length=1)

    generation_config_sha256: str

    repetitions: int = Field(..., ge=1)

    members: tuple[FingerprintReferenceSetMember, ...]

    content_sha256: str

    model_config = {"frozen": True, "extra": "forbid"}

    @field_validator("suite_content_sha256", "generation_config_sha256", "content_sha256")
    @classmethod
    def _validate_sha256(cls, v: str, info: ValidationInfo) -> str:
        field_name = info.field_name
        assert field_name is not None
        return normalize_sha256(v, field_name)

    @model_validator(mode="after")
    def _validate_members(self) -> FingerprintReferenceSet:
        if not self.members:
            raise ValueError("a fingerprint reference set must contain at least one member")
        ids = [member.snapshot_id for member in self.members]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate snapshot_id in fingerprint reference set members: {ids}")
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
            raise FingerprintReferenceSetIntegrityError(
                f"fingerprint reference set '{self.fingerprint_set_id}' v{self.fingerprint_set_version} "
                f"content hash mismatch: recomputed {actual!r} != declared {self.content_sha256!r}"
            )
        return actual


class FingerprintReferenceSetBuilder:
    """兼容门禁（Step 14.1）下的集合构建器.

    ``allow_test_fixture`` 仅供测试内部 helper 使用；production 路径与 CLI
    永远不传递它（不提供 ``--allow-test-fixture``）。
    """

    def __init__(self, *, allow_test_fixture: bool = False) -> None:
        self._allow_test_fixture = allow_test_fixture

    def build(
        self,
        *,
        fingerprint_set_id: str,
        fingerprint_set_version: str,
        snapshots: Sequence[FingerprintReferenceSnapshot],
        snapshot_sha256s: Mapping[str, str],
    ) -> FingerprintReferenceSet:
        """由已验证成员快照装配集合.

        Args:
            snapshots: 成员快照（套件 / 策略 / 生成配置 / 重复轮数必须一致）。
            snapshot_sha256s: ``{snapshot_id: sha256_of_persisted_file_bytes}``，
                必须覆盖每个成员（调用方先做磁盘字节校验）。

        Raises:
            FixtureFingerprintReferenceError: 成员是 ``test_fixture`` 且未开例外。
            FingerprintReferenceSetCompatibilityError: 任一成员未通过门禁。
            FingerprintReferenceSetError: 空集合、重复 id、缺少已验证摘要。
        """
        member_snapshots = list(snapshots)
        if not member_snapshots:
            raise FingerprintReferenceSetError("cannot build a fingerprint reference set with no member snapshots")

        for snapshot in member_snapshots:
            if snapshot.source_role == FingerprintSourceRole.CANDIDATE_CAPTURE:
                raise FingerprintReferenceSetError(
                    f"fingerprint snapshot '{snapshot.snapshot_id}' is a candidate capture from an audited "
                    f"endpoint; candidate captures are never reference material and can never enter a "
                    f"fingerprint reference set"
                )
            if snapshot.source_role == FingerprintSourceRole.TEST_FIXTURE and not self._allow_test_fixture:
                raise FixtureFingerprintReferenceError(
                    f"fingerprint snapshot '{snapshot.snapshot_id}' is a test_fixture; "
                    f"test fixtures never enter a trusted fingerprint reference set"
                )

        ids = [snapshot.snapshot_id for snapshot in member_snapshots]
        if len(set(ids)) != len(ids):
            raise FingerprintReferenceSetError(f"duplicate snapshot_id in fingerprint reference set members: {ids}")
        missing = [snapshot_id for snapshot_id in ids if snapshot_id not in snapshot_sha256s]
        if missing:
            raise FingerprintReferenceSetError(f"missing verified snapshot SHA-256 for member(s): {missing}")

        # 确定性成员顺序，保证 content hash 稳定。
        ordered = sorted(member_snapshots, key=lambda snapshot: snapshot.snapshot_id)
        base = ordered[0]
        for snapshot in ordered[1:]:
            self._assert_compatible(base, snapshot)

        members = tuple(
            FingerprintReferenceSetMember(
                snapshot_id=snapshot.snapshot_id,
                snapshot_sha256=snapshot_sha256s[snapshot.snapshot_id],
                model_id=snapshot.model_id,
                provider_id=snapshot.provider_id,
                source_role=snapshot.source_role,
            )
            for snapshot in ordered
        )

        payload: dict[str, Any] = {
            "fingerprint_set_id": fingerprint_set_id,
            "fingerprint_set_version": fingerprint_set_version,
            "suite_id": base.suite_id,
            "suite_version": base.suite_version,
            "suite_content_sha256": base.suite_content_sha256,
            "normalization_policy_id": base.normalization_policy_id,
            "normalization_policy_version": base.normalization_policy_version,
            "generation_config_sha256": base.generation_config_sha256,
            "repetitions": base.repetitions,
            "members": [member.model_dump() for member in members],
        }
        provisional = FingerprintReferenceSet(**payload, content_sha256="0" * 64)
        fingerprint_set = provisional.model_copy(update={"content_sha256": provisional.compute_content_sha256()})
        fingerprint_set.verify_content_hash()
        return fingerprint_set

    @staticmethod
    def _assert_compatible(
        base: FingerprintReferenceSnapshot,
        other: FingerprintReferenceSnapshot,
    ) -> None:
        """Step 14.1 的 fail-closed 兼容门禁."""
        mismatches: list[str] = []

        def check(name: str, actual: object, expected: object) -> None:
            if actual != expected:
                mismatches.append(f"{name}: {actual!r} != {expected!r}")

        check("suite_id", other.suite_id, base.suite_id)
        check("suite_version", other.suite_version, base.suite_version)
        check("suite_content_sha256", other.suite_content_sha256, base.suite_content_sha256)
        check("normalization_policy_id", other.normalization_policy_id, base.normalization_policy_id)
        check(
            "normalization_policy_version",
            other.normalization_policy_version,
            base.normalization_policy_version,
        )
        check("generation_config_sha256", other.generation_config_sha256, base.generation_config_sha256)
        check("repetitions", other.repetitions, base.repetitions)

        if mismatches:
            raise FingerprintReferenceSetCompatibilityError(
                f"fingerprint snapshot '{other.snapshot_id}' is incompatible with "
                f"'{base.snapshot_id}': " + "; ".join(mismatches)
            )


__all__: list[str] = [
    "FingerprintReferenceSetError",
    "DuplicateFingerprintReferenceSetError",
    "FingerprintReferenceSetNotFoundError",
    "FingerprintReferenceSetIntegrityError",
    "FingerprintReferenceSetCompatibilityError",
    "FixtureFingerprintReferenceError",
    "FingerprintReferenceSetMember",
    "FingerprintReferenceSet",
    "FingerprintReferenceSetBuilder",
]
