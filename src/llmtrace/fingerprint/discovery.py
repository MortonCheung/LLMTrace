"""Fingerprint reference set auto-discovery（Task 36）.

用户把受信任的指纹参考集放在默认位置（``~/.llmtrace/fingerprints/sets/``）时，
不应每次运行都手写 ``--fingerprint-set``。本模块扫描该目录，只保留真正能通过
:func:`resolve_fingerprint_context` 的集合（套件身份 + 重复轮数 + 成员快照
完整性 + 自哈希全部一致）。

优先级（Task 36）::

    explicit --fingerprint-set  >  唯一 compatible auto-discovery  >  none

规则：

* 0 个 compatible → 指纹不可用，run 继续（只是没有身份证据）
* 恰好 1 个      → 自动使用，并在 run 开始前告知
* >1 个          → 绝不随机挑一个；非交互环境 fail closed，要求显式指定

本模块全程只读：不发目标 HTTP、不读 API key、不建 provider、不写 artifact。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llmtrace.fingerprint.repository import FingerprintRepository
from llmtrace.fingerprint.runtime import resolve_fingerprint_context


@dataclass(frozen=True)
class FingerprintDiscoveryResolution:
    """一次 run 的指纹参考集解析结果（Task 36）.

    ``resolved_path`` 仅在确实找到唯一兼容集合（显式或发现）时非空；
    ``candidates`` 在存在多个兼容集合时携带全部候选，供 CLI fail closed。
    """

    resolved_path: Path | None
    candidates: tuple[Path, ...] = ()


def discover_compatible_sets(
    repository: FingerprintRepository,
    *,
    repetitions: int,
    sets_dir: Path | None = None,
) -> list[Path]:
    """返回 ``sets_dir`` 下与本次运行兼容的参考集路径（按路径排序，确定）.

    每个候选都要完整通过 :func:`resolve_fingerprint_context`（自哈希、套件身份、
    重复轮数、成员快照完整性、聚合）；任何一项失败即被跳过，而不是致命错误。
    """
    if repetitions < 1:
        raise ValueError(f"repetitions must be >= 1, got {repetitions}")
    directory = sets_dir if sets_dir is not None else repository.layout.fingerprint_sets_dir
    if not directory.is_dir():
        return []
    candidates: list[Path] = []
    for set_path in sorted(directory.glob("*.json")):
        try:
            resolve_fingerprint_context(set_path=set_path, repository=repository, repetitions=repetitions)
        except Exception:
            # 不可解析 / 不可信 / 与本次轮数不兼容的集合绝不参与自动发现。
            continue
        candidates.append(set_path)
    return candidates


def resolve_fingerprint_set(
    *,
    explicit_path: Path | None,
    repository: FingerprintRepository,
    repetitions: int,
    sets_dir: Path | None = None,
) -> FingerprintDiscoveryResolution:
    """解析本次运行的指纹参考集：explicit > 唯一发现 > none（Task 36）.

    Raises:
        FingerprintRuntimeError: 显式路径无法通过校验（与 runner 预检同一门禁）。
        ValueError: repetitions < 1。
    """
    if explicit_path is not None:
        resolve_fingerprint_context(set_path=explicit_path, repository=repository, repetitions=repetitions)
        return FingerprintDiscoveryResolution(resolved_path=explicit_path)

    candidates = discover_compatible_sets(repository, repetitions=repetitions, sets_dir=sets_dir)
    if len(candidates) == 1:
        return FingerprintDiscoveryResolution(resolved_path=candidates[0], candidates=(candidates[0],))
    return FingerprintDiscoveryResolution(resolved_path=None, candidates=tuple(candidates))


__all__: list[str] = [
    "FingerprintDiscoveryResolution",
    "discover_compatible_sets",
    "resolve_fingerprint_set",
]
