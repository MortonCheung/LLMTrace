"""Fingerprint repository（Task 13）.

默认目录来自 ``appdir.py``（不硬编码 home path）：::

    ~/.llmtrace/fingerprints/
    ├── snapshots/   # FingerprintReferenceSnapshot
    ├── sets/        # FingerprintReferenceSet
    └── policies/    # FingerprintDecisionPolicy（append-only，Task 19）

落盘纪律与既有 ``scoring/reference.py`` / ``reference/repository.py`` 一致：

1. 内存索引先查重；
2. 持久化时用 ``"x"`` 独占创建，任何进程看到同名文件都视为重复而不是覆盖；
3. 磁盘写入成功之后才登记到内存，失败的写入不会污染索引。

本模块只是存储层：不计算兼容性、不改写内容身份。内容哈希由模型自校验
（``verify_content_hash``），兼容门禁由 ``reference_set`` 的 builder 负责。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from llmtrace.appdir import AppLayout, ensure_app_layout
from llmtrace.fingerprint.policy import (
    FingerprintDecisionPolicy,
    FingerprintPolicyIntegrityError,
)
from llmtrace.fingerprint.reference import FingerprintReferenceSnapshot
from llmtrace.fingerprint.reference_set import (
    DuplicateFingerprintReferenceSetError,
    FingerprintReferenceSet,
    FingerprintReferenceSetIntegrityError,
    FingerprintReferenceSetNotFoundError,
)
from llmtrace.utilities.hashing import sha256_hash as sha256_of


class FingerprintRepositoryError(Exception):
    """Fingerprint 存储层错误基类."""

    error_code = "FINGERPRINT_REPOSITORY_ERROR"


class DuplicateFingerprintSnapshotError(FingerprintRepositoryError):
    """同 ``snapshot_id`` 已存在；snapshot 不可变且 append-only."""


class FingerprintSnapshotNotFoundError(FingerprintRepositoryError):
    """请求的 ``snapshot_id`` 不存在."""


class FingerprintSnapshotIntegrityError(FingerprintRepositoryError):
    """磁盘字节与声明的内容身份不一致."""


class DuplicateFingerprintPolicyError(FingerprintRepositoryError):
    """同 ``(policy_id, policy_version)`` 已存在；policy 不可变且 append-only."""


class FingerprintPolicyNotFoundError(FingerprintRepositoryError):
    """请求的 ``(policy_id, policy_version)`` 不存在."""


def _resolve_child(base: Path, filename: str, label: str) -> Path:
    """把 ``filename`` 解析为 ``base`` 的直接子路径，越界即失败.

    纵深防御：即使 id 校验器将来被放宽，落盘目标也必须留在仓库目录内。
    """
    resolved_base = base.resolve()
    target = (resolved_base / filename).resolve()
    if target.parent != resolved_base:
        raise ValueError(f"{label} resolves outside the repository directory '{resolved_base}'")
    return target


class FingerprintSnapshotRepository:
    """``FingerprintReferenceSnapshot`` 的 append-only JSON 存储.

    ``directory=None`` 表示纯内存仓库（测试用）；生产路径用
    ``default()`` / ``load_default()`` 取 appdir 下的 ``fingerprints/snapshots``。
    """

    def __init__(self, *, directory: Path | None = None) -> None:
        self._directory = directory
        self._snapshots: dict[str, FingerprintReferenceSnapshot] = {}

    # -- Write -------------------------------------------------------------

    def save(self, snapshot: FingerprintReferenceSnapshot) -> FingerprintReferenceSnapshot:
        """存储 *snapshot*，拒绝已存在的 ``snapshot_id``.

        Raises:
            DuplicateFingerprintSnapshotError: 内存索引或磁盘上已存在该 id。
        """
        key = snapshot.snapshot_id
        if key in self._snapshots:
            raise DuplicateFingerprintSnapshotError(
                f"fingerprint snapshot '{key}' already exists; snapshots are immutable and append-only — "
                f"use a new snapshot_id for a new capture"
            )

        # 先磁盘后内存：失败的写入不得污染索引。
        if self._directory is not None:
            self._write_file(snapshot)

        self._snapshots[key] = snapshot
        return snapshot

    # -- Read --------------------------------------------------------------

    def get(self, snapshot_id: str) -> FingerprintReferenceSnapshot:
        """返回 *snapshot_id* 对应的快照.

        Raises:
            FingerprintSnapshotNotFoundError: 该 id 不存在。
        """
        try:
            return self._snapshots[snapshot_id]
        except KeyError as exc:
            raise FingerprintSnapshotNotFoundError(f"fingerprint snapshot '{snapshot_id}' not found") from exc

    def read_raw(self, snapshot_id: str) -> str:
        """返回已落盘快照文件的原始 JSON 文本（只读，不写盘）."""
        if self._directory is None:
            raise FingerprintSnapshotNotFoundError(
                f"fingerprint snapshot '{snapshot_id}' has no persisted file (in-memory repository)"
            )
        path = self._file_path(snapshot_id)
        if not path.exists():
            raise FingerprintSnapshotNotFoundError(
                f"fingerprint snapshot '{snapshot_id}' not found on disk at '{path}'"
            )
        return path.read_text(encoding="utf-8")

    def snapshot_sha256(self, snapshot_id: str) -> str:
        """返回 *snapshot_id* 已落盘文件字节的 SHA-256（供 reference set 成员引用）."""
        return sha256_of(self.read_raw(snapshot_id))

    def verify(self, snapshot_id: str, expected_sha256: str | None = None) -> str:
        """校验快照完整性，任何不一致都 fail closed.

        1. 重算的内容哈希必须等于声明的 ``content_sha256``；
        2. 已落盘时，磁盘字节摘要必须等于 *expected_sha256*（缺省则与内存序列化比对）。

        Returns:
            已落盘时返回磁盘文件 SHA-256，否则返回已验证的内容哈希。

        Raises:
            FingerprintSnapshotIntegrityError: 任一哈希不一致。
        """
        snapshot = self.get(snapshot_id)
        snapshot.verify_content_hash()

        if self._directory is not None:
            actual_disk = self.snapshot_sha256(snapshot_id)
            expected = expected_sha256 or sha256_of(snapshot.model_dump_json(indent=2))
            if actual_disk != expected:
                raise FingerprintSnapshotIntegrityError(
                    f"fingerprint snapshot '{snapshot_id}' on-disk integrity check failed: "
                    f"file SHA-256 {actual_disk!r} != expected {expected!r}"
                )
            return actual_disk
        return snapshot.content_sha256

    def list(self) -> Sequence[FingerprintReferenceSnapshot]:
        """按 ``snapshot_id`` 排序返回全部快照."""
        return sorted(self._snapshots.values(), key=lambda snapshot: snapshot.snapshot_id)

    def __len__(self) -> int:
        return len(self._snapshots)

    def __contains__(self, snapshot_id: str) -> bool:
        return snapshot_id in self._snapshots

    @classmethod
    def load(cls, directory: Path) -> FingerprintSnapshotRepository:
        """加载 *directory* 下全部 ``*.json`` 快照."""
        repository = cls(directory=directory)
        repository._load_directory(directory)
        return repository

    @classmethod
    def default(cls, *, data_root: Path | None = None) -> FingerprintSnapshotRepository:
        """使用 appdir 默认目录（``fingerprints/snapshots``）的仓库."""
        return cls(directory=ensure_app_layout(data_root).fingerprint_snapshots_dir)

    @classmethod
    def load_default(cls, *, data_root: Path | None = None) -> FingerprintSnapshotRepository:
        """使用 appdir 默认目录并加载既有快照."""
        layout = ensure_app_layout(data_root)
        return cls.load(layout.fingerprint_snapshots_dir)

    # -- File I/O ----------------------------------------------------------

    def _file_path(self, snapshot_id: str) -> Path:
        assert self._directory is not None
        return _resolve_child(self._directory, f"{snapshot_id}.json", f"fingerprint snapshot id '{snapshot_id}'")

    def _write_file(self, snapshot: FingerprintReferenceSnapshot) -> None:
        """独占创建快照文件；已存在即视为重复（append-only）."""
        assert self._directory is not None
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._file_path(snapshot.snapshot_id)
        try:
            with path.open("x", encoding="utf-8") as f:
                f.write(snapshot.model_dump_json(indent=2))
        except FileExistsError as exc:
            raise DuplicateFingerprintSnapshotError(
                f"fingerprint snapshot '{snapshot.snapshot_id}' already exists on disk at '{path}'; "
                f"snapshots are immutable and append-only — use a new snapshot_id for a new capture"
            ) from exc

    def _load_directory(self, directory: Path) -> None:
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.json")):
            snapshot = FingerprintReferenceSnapshot.model_validate(json.loads(path.read_text(encoding="utf-8")))
            key = snapshot.snapshot_id
            if key in self._snapshots:
                raise DuplicateFingerprintSnapshotError(
                    f"duplicate fingerprint snapshot '{key}' found in '{directory}'"
                )
            self._snapshots[key] = snapshot


class FingerprintReferenceSetRepository:
    """``FingerprintReferenceSet`` 的 append-only JSON 存储.

    键为 ``(fingerprint_set_id, fingerprint_set_version)``；新修订必须使用新
    version（集合不可变）。``directory=None`` 表示纯内存仓库（测试用）。
    """

    def __init__(self, *, directory: Path | None = None) -> None:
        self._directory = directory
        self._sets: dict[tuple[str, str], FingerprintReferenceSet] = {}

    # -- Write -------------------------------------------------------------

    def save(self, fingerprint_set: FingerprintReferenceSet) -> FingerprintReferenceSet:
        """存储 *fingerprint_set*，拒绝已存在的 ``(id, version)``.

        Raises:
            DuplicateFingerprintReferenceSetError: 内存索引或磁盘上已存在该键。
        """
        key = (fingerprint_set.fingerprint_set_id, fingerprint_set.fingerprint_set_version)
        if key in self._sets:
            raise DuplicateFingerprintReferenceSetError(
                f"fingerprint reference set '{key[0]}' v{key[1]} already exists; sets are immutable and "
                f"append-only — use a new fingerprint_set_version for a new revision"
            )

        # 先磁盘后内存：失败的写入不得污染索引。
        if self._directory is not None:
            self._write_file(fingerprint_set)

        self._sets[key] = fingerprint_set
        return fingerprint_set

    # -- Read --------------------------------------------------------------

    def get(self, fingerprint_set_id: str, fingerprint_set_version: str) -> FingerprintReferenceSet:
        """返回 ``(fingerprint_set_id, fingerprint_set_version)`` 对应的集合.

        Raises:
            FingerprintReferenceSetNotFoundError: 该键不存在。
        """
        try:
            return self._sets[(fingerprint_set_id, fingerprint_set_version)]
        except KeyError as exc:
            raise FingerprintReferenceSetNotFoundError(
                f"fingerprint reference set '{fingerprint_set_id}' v{fingerprint_set_version} not found"
            ) from exc

    def read_raw(self, fingerprint_set_id: str, fingerprint_set_version: str) -> str:
        """返回已落盘集合文件的原始 JSON 文本（只读，不写盘）."""
        if self._directory is None:
            raise FingerprintReferenceSetNotFoundError(
                f"fingerprint reference set '{fingerprint_set_id}' v{fingerprint_set_version} "
                f"has no persisted file (in-memory repository)"
            )
        path = self._file_path(fingerprint_set_id, fingerprint_set_version)
        if not path.exists():
            raise FingerprintReferenceSetNotFoundError(
                f"fingerprint reference set '{fingerprint_set_id}' v{fingerprint_set_version} "
                f"not found on disk at '{path}'"
            )
        return path.read_text(encoding="utf-8")

    def set_sha256(self, fingerprint_set_id: str, fingerprint_set_version: str) -> str:
        """返回已落盘集合文件字节的 SHA-256."""
        return sha256_of(self.read_raw(fingerprint_set_id, fingerprint_set_version))

    def verify(
        self,
        fingerprint_set_id: str,
        fingerprint_set_version: str,
        expected_sha256: str | None = None,
    ) -> str:
        """校验集合完整性，任何不一致都 fail closed.

        Raises:
            FingerprintReferenceSetIntegrityError: 任一哈希不一致。
        """
        fingerprint_set = self.get(fingerprint_set_id, fingerprint_set_version)
        fingerprint_set.verify_content_hash()

        if self._directory is not None:
            actual_disk = self.set_sha256(fingerprint_set_id, fingerprint_set_version)
            expected = expected_sha256 or sha256_of(fingerprint_set.model_dump_json(indent=2))
            if actual_disk != expected:
                raise FingerprintReferenceSetIntegrityError(
                    f"fingerprint reference set '{fingerprint_set_id}' v{fingerprint_set_version} "
                    f"on-disk integrity check failed: file SHA-256 {actual_disk!r} != expected {expected!r}"
                )
            return actual_disk
        return fingerprint_set.content_sha256

    def list(self) -> Sequence[FingerprintReferenceSet]:
        """按 ``(fingerprint_set_id, fingerprint_set_version)`` 排序返回全部集合."""
        return sorted(
            self._sets.values(),
            key=lambda item: (item.fingerprint_set_id, item.fingerprint_set_version),
        )

    def __len__(self) -> int:
        return len(self._sets)

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self._sets

    @classmethod
    def load(cls, directory: Path) -> FingerprintReferenceSetRepository:
        """加载 *directory* 下全部 ``*.json`` 集合."""
        repository = cls(directory=directory)
        repository._load_directory(directory)
        return repository

    @classmethod
    def default(cls, *, data_root: Path | None = None) -> FingerprintReferenceSetRepository:
        """使用 appdir 默认目录（``fingerprints/sets``）的仓库."""
        return cls(directory=ensure_app_layout(data_root).fingerprint_sets_dir)

    @classmethod
    def load_default(cls, *, data_root: Path | None = None) -> FingerprintReferenceSetRepository:
        """使用 appdir 默认目录并加载既有集合."""
        layout = ensure_app_layout(data_root)
        return cls.load(layout.fingerprint_sets_dir)

    # -- File I/O ----------------------------------------------------------

    def _file_path(self, fingerprint_set_id: str, fingerprint_set_version: str) -> Path:
        assert self._directory is not None
        return _resolve_child(
            self._directory,
            f"{fingerprint_set_id}_{fingerprint_set_version}.json",
            f"fingerprint set id '{fingerprint_set_id}' v{fingerprint_set_version}",
        )

    def _write_file(self, fingerprint_set: FingerprintReferenceSet) -> None:
        """独占创建集合文件；已存在即视为重复（append-only）."""
        assert self._directory is not None
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._file_path(fingerprint_set.fingerprint_set_id, fingerprint_set.fingerprint_set_version)
        try:
            with path.open("x", encoding="utf-8") as f:
                f.write(fingerprint_set.model_dump_json(indent=2))
        except FileExistsError as exc:
            raise DuplicateFingerprintReferenceSetError(
                f"fingerprint reference set '{fingerprint_set.fingerprint_set_id}' "
                f"v{fingerprint_set.fingerprint_set_version} already exists on disk at '{path}'; "
                f"sets are immutable and append-only — use a new fingerprint_set_version for a new revision"
            ) from exc

    def _load_directory(self, directory: Path) -> None:
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.json")):
            fingerprint_set = FingerprintReferenceSet.model_validate(json.loads(path.read_text(encoding="utf-8")))
            key = (fingerprint_set.fingerprint_set_id, fingerprint_set.fingerprint_set_version)
            if key in self._sets:
                raise DuplicateFingerprintReferenceSetError(
                    f"duplicate fingerprint reference set '{key[0]}' v{key[1]} found in '{directory}'"
                )
            self._sets[key] = fingerprint_set


class FingerprintDecisionPolicyRepository:
    """``FingerprintDecisionPolicy`` 的 append-only JSON 存储（Task 19）.

    键为 ``(policy_id, policy_version)``；新修订必须使用新 version。已验证的
    policy 一旦落盘就不可被"重新验证"覆盖，避免历史判定漂移。
    ``directory=None`` 表示纯内存仓库（测试用）。
    """

    def __init__(self, *, directory: Path | None = None) -> None:
        self._directory = directory
        self._policies: dict[tuple[str, str], FingerprintDecisionPolicy] = {}

    # -- Write -------------------------------------------------------------

    def save(self, policy: FingerprintDecisionPolicy) -> FingerprintDecisionPolicy:
        """存储 *policy*，拒绝已存在的 ``(policy_id, policy_version)``.

        Raises:
            DuplicateFingerprintPolicyError: 内存索引或磁盘上已存在该键。
        """
        key = (policy.policy_id, policy.policy_version)
        if key in self._policies:
            raise DuplicateFingerprintPolicyError(
                f"fingerprint policy '{key[0]}' v{key[1]} already exists; policies are immutable and "
                f"append-only — use a new policy_version for a new revision"
            )

        # 先磁盘后内存：失败的写入不得污染索引。
        if self._directory is not None:
            self._write_file(policy)

        self._policies[key] = policy
        return policy

    # -- Read --------------------------------------------------------------

    def get(self, policy_id: str, policy_version: str) -> FingerprintDecisionPolicy:
        """返回 ``(policy_id, policy_version)`` 对应的 policy.

        Raises:
            FingerprintPolicyNotFoundError: 该键不存在。
        """
        try:
            return self._policies[(policy_id, policy_version)]
        except KeyError as exc:
            raise FingerprintPolicyNotFoundError(
                f"fingerprint policy '{policy_id}' v{policy_version} not found"
            ) from exc

    def read_raw(self, policy_id: str, policy_version: str) -> str:
        """返回已落盘 policy 文件的原始 JSON 文本（只读，不写盘）."""
        if self._directory is None:
            raise FingerprintPolicyNotFoundError(
                f"fingerprint policy '{policy_id}' v{policy_version} has no persisted file (in-memory repository)"
            )
        path = self._file_path(policy_id, policy_version)
        if not path.exists():
            raise FingerprintPolicyNotFoundError(
                f"fingerprint policy '{policy_id}' v{policy_version} not found on disk at '{path}'"
            )
        return path.read_text(encoding="utf-8")

    def policy_sha256(self, policy_id: str, policy_version: str) -> str:
        """返回已落盘 policy 文件字节的 SHA-256."""
        return sha256_of(self.read_raw(policy_id, policy_version))

    def verify(self, policy_id: str, policy_version: str, expected_sha256: str | None = None) -> str:
        """校验 policy 完整性，任何不一致都 fail closed.

        Raises:
            FingerprintPolicyIntegrityError: 任一哈希不一致。
        """
        policy = self.get(policy_id, policy_version)
        policy.verify_content_hash()

        if self._directory is not None:
            actual_disk = self.policy_sha256(policy_id, policy_version)
            expected = expected_sha256 or sha256_of(policy.model_dump_json(indent=2))
            if actual_disk != expected:
                raise FingerprintPolicyIntegrityError(
                    f"fingerprint policy '{policy_id}' v{policy_version} on-disk integrity check failed: "
                    f"file SHA-256 {actual_disk!r} != expected {expected!r}"
                )
            return actual_disk
        return policy.content_sha256

    def list(self) -> Sequence[FingerprintDecisionPolicy]:
        """按 ``(policy_id, policy_version)`` 排序返回全部 policy."""
        return sorted(self._policies.values(), key=lambda item: (item.policy_id, item.policy_version))

    def __len__(self) -> int:
        return len(self._policies)

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self._policies

    @classmethod
    def load(cls, directory: Path) -> FingerprintDecisionPolicyRepository:
        """加载 *directory* 下全部 ``*.json`` policy."""
        repository = cls(directory=directory)
        repository._load_directory(directory)
        return repository

    @classmethod
    def default(cls, *, data_root: Path | None = None) -> FingerprintDecisionPolicyRepository:
        """使用 appdir 默认目录（``fingerprints/policies``）的仓库."""
        return cls(directory=ensure_app_layout(data_root).fingerprint_policies_dir)

    @classmethod
    def load_default(cls, *, data_root: Path | None = None) -> FingerprintDecisionPolicyRepository:
        """使用 appdir 默认目录并加载既有 policy."""
        layout = ensure_app_layout(data_root)
        return cls.load(layout.fingerprint_policies_dir)

    # -- File I/O ----------------------------------------------------------

    def _file_path(self, policy_id: str, policy_version: str) -> Path:
        assert self._directory is not None
        return _resolve_child(
            self._directory,
            f"{policy_id}_{policy_version}.json",
            f"fingerprint policy id '{policy_id}' v{policy_version}",
        )

    def _write_file(self, policy: FingerprintDecisionPolicy) -> None:
        """独占创建 policy 文件；已存在即视为重复（append-only）."""
        assert self._directory is not None
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._file_path(policy.policy_id, policy.policy_version)
        try:
            with path.open("x", encoding="utf-8") as f:
                f.write(policy.model_dump_json(indent=2))
        except FileExistsError as exc:
            raise DuplicateFingerprintPolicyError(
                f"fingerprint policy '{policy.policy_id}' v{policy.policy_version} already exists on disk at "
                f"'{path}'; policies are immutable and append-only — use a new policy_version for a new revision"
            ) from exc

    def _load_directory(self, directory: Path) -> None:
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.json")):
            policy = FingerprintDecisionPolicy.model_validate(json.loads(path.read_text(encoding="utf-8")))
            key = (policy.policy_id, policy.policy_version)
            if key in self._policies:
                raise DuplicateFingerprintPolicyError(
                    f"duplicate fingerprint policy '{key[0]}' v{key[1]} found in '{directory}'"
                )
            self._policies[key] = policy


class FingerprintRepository:
    """appdir 布局下的 fingerprint 仓库根（Task 13）."""

    def __init__(self, *, data_root: Path | None = None) -> None:
        self.layout: AppLayout = ensure_app_layout(data_root)
        self.snapshots = FingerprintSnapshotRepository(directory=self.layout.fingerprint_snapshots_dir)
        self.sets = FingerprintReferenceSetRepository(directory=self.layout.fingerprint_sets_dir)
        self.policies = FingerprintDecisionPolicyRepository(directory=self.layout.fingerprint_policies_dir)

    @property
    def policies_dir(self) -> Path:
        return self.layout.fingerprint_policies_dir

    @classmethod
    def load(cls, *, data_root: Path | None = None) -> FingerprintRepository:
        """构建仓库并加载磁盘上已有的快照、集合与 policy（append-only，不覆盖）."""
        repository = cls(data_root=data_root)
        repository.snapshots = FingerprintSnapshotRepository.load(repository.layout.fingerprint_snapshots_dir)
        repository.sets = FingerprintReferenceSetRepository.load(repository.layout.fingerprint_sets_dir)
        repository.policies = FingerprintDecisionPolicyRepository.load(repository.layout.fingerprint_policies_dir)
        return repository

    @classmethod
    def in_memory(cls) -> FingerprintRepository:
        """纯内存仓库（测试用，不触碰磁盘）."""
        repository = cls.__new__(cls)
        repository.layout = AppLayout.resolve()
        repository.snapshots = FingerprintSnapshotRepository()
        repository.sets = FingerprintReferenceSetRepository()
        repository.policies = FingerprintDecisionPolicyRepository()
        return repository


__all__: list[str] = [
    "FingerprintRepositoryError",
    "DuplicateFingerprintSnapshotError",
    "FingerprintSnapshotNotFoundError",
    "FingerprintSnapshotIntegrityError",
    "DuplicateFingerprintPolicyError",
    "FingerprintPolicyNotFoundError",
    "FingerprintSnapshotRepository",
    "FingerprintReferenceSetRepository",
    "FingerprintDecisionPolicyRepository",
    "FingerprintRepository",
]
