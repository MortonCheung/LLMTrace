"""Fingerprint suite loading / validation / content identity（Task 5）。

套件是 data-driven 的：探测项只存在于 ``resources/fingerprint_v1.json``，
不允许把 probe 字面量散落在 executor 里。

内容身份（``content_sha256``）不是对原始 JSON 文件字节取哈希，而是对**规范化
后的语义载荷**取 canonical JSON + SHA-256 —— 与 Quick Suite 的既有做法一致，
因此空白/键序等修饰性变化不改变身份，而探测内容变化必然改变身份。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from llmtrace.fingerprint.models import (
    FingerprintProbe,
    FingerprintSuite,
    normalize_sha256,
)
from llmtrace.fingerprint.normalizers import resolve_normalizer
from llmtrace.utilities.hashing import canonical_json_hash

_RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
DEFAULT_SUITE_FILENAME = "fingerprint_v1.json"

_ALLOWED_SUITE_KEYS = frozenset(
    {
        "suite_id",
        "suite_version",
        "normalization_policy_id",
        "normalization_policy_version",
        "probes",
        "content_sha256",
    }
)


class FingerprintSuiteError(Exception):
    """Fingerprint suite 错误基类."""

    error_code = "FINGERPRINT_SUITE_ERROR"


class FingerprintSuiteNotFoundError(FingerprintSuiteError):
    """套件文件不存在."""


class FingerprintSuiteValidationError(FingerprintSuiteError):
    """套件结构或内容不合法（fail closed）."""


class FingerprintSuiteIntegrityError(FingerprintSuiteError):
    """套件声明的 content_sha256 与实际内容不一致."""


def default_fingerprint_suite_path() -> Path:
    """内置套件的真实路径."""
    return _RESOURCE_DIR / DEFAULT_SUITE_FILENAME


def suite_content_payload(suite: FingerprintSuite) -> dict[str, Any]:
    """套件的规范语义载荷（probe 按声明顺序，字段由模型定义穷举）."""
    return {
        "suite_id": suite.suite_id,
        "suite_version": suite.suite_version,
        "normalization_policy_id": suite.normalization_policy_id,
        "normalization_policy_version": suite.normalization_policy_version,
        "probes": [probe.model_dump(mode="json") for probe in suite.probes],
    }


def compute_suite_content_sha256(suite: FingerprintSuite) -> str:
    """套件内容身份 = canonical JSON(语义载荷) 的 SHA-256."""
    return canonical_json_hash(suite_content_payload(suite))


def compute_generation_config_sha256(suite: FingerprintSuite) -> str:
    """生成配置身份：每个 probe 的 (probe_id, temperature, max_output_tokens) 规范哈希.

    该身份用于参考集兼容门禁：温度或输出上限不同的两次采集不可直接比较。
    """
    payload = {
        "probes": [
            {
                "probe_id": probe.probe_id,
                "temperature": probe.temperature,
                "max_output_tokens": probe.max_output_tokens,
            }
            for probe in suite.probes
        ]
    }
    return canonical_json_hash(payload)


def build_fingerprint_suite(payload: Mapping[str, Any], *, source: str = "<memory>") -> FingerprintSuite:
    """从 JSON 载荷构造并自校验一个 ``FingerprintSuite``。

    Raises:
        FingerprintSuiteValidationError: 结构非法、未知字段、重复 probe、
            或归一化策略不受支持。
        FingerprintSuiteIntegrityError: 载荷自带 ``content_sha256`` 且与内容不符。
    """
    unknown = set(payload.keys()) - _ALLOWED_SUITE_KEYS
    if unknown:
        raise FingerprintSuiteValidationError(f"suite {source} has unknown top-level keys: {sorted(unknown)}")

    probes_payload = payload.get("probes")
    if not isinstance(probes_payload, list) or not probes_payload:
        raise FingerprintSuiteValidationError(f"suite {source} must declare a non-empty 'probes' list")

    probes: list[FingerprintProbe] = []
    for index, raw_probe in enumerate(probes_payload):
        if not isinstance(raw_probe, dict):
            raise FingerprintSuiteValidationError(f"suite {source} probe #{index} must be an object")
        try:
            probes.append(FingerprintProbe.model_validate(raw_probe))
        except Exception as exc:  # pydantic ValidationError → 统一 fail closed
            raise FingerprintSuiteValidationError(f"suite {source} probe #{index} is invalid: {exc}") from exc

    def _require_str(key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise FingerprintSuiteValidationError(f"suite {source} field {key!r} must be a non-empty string")
        return value

    suite_id = _require_str("suite_id")
    suite_version = _require_str("suite_version")
    normalization_policy_id = _require_str("normalization_policy_id")
    normalization_policy_version = _require_str("normalization_policy_version")

    # 未知归一化策略在加载期即 fail closed，绝不进入执行阶段。
    resolve_normalizer(normalization_policy_id, normalization_policy_version)

    # 先算身份，再构造模型（模型要求 content_sha256 非空且为合法 SHA-256）。
    probe_ids = [probe.probe_id for probe in probes]
    if len(set(probe_ids)) != len(probe_ids):
        raise FingerprintSuiteValidationError(f"suite {source} has duplicate probe_id: {probe_ids}")

    provisional = FingerprintSuite.model_construct(
        suite_id=suite_id,
        suite_version=suite_version,
        normalization_policy_id=normalization_policy_id,
        normalization_policy_version=normalization_policy_version,
        probes=tuple(probes),
        content_sha256="0" * 64,
    )
    content_sha256 = compute_suite_content_sha256(provisional)

    declared = payload.get("content_sha256")
    if declared is not None:
        if not isinstance(declared, str):
            raise FingerprintSuiteValidationError(f"suite {source} content_sha256 must be a string")
        if normalize_sha256(declared, "content_sha256") != content_sha256:
            raise FingerprintSuiteIntegrityError(
                f"suite {source} content_sha256 mismatch: declared {declared!r} != computed {content_sha256!r}"
            )

    try:
        return FingerprintSuite(
            suite_id=suite_id,
            suite_version=suite_version,
            normalization_policy_id=normalization_policy_id,
            normalization_policy_version=normalization_policy_version,
            probes=tuple(probes),
            content_sha256=content_sha256,
        )
    except Exception as exc:
        raise FingerprintSuiteValidationError(f"suite {source} is invalid: {exc}") from exc


def load_fingerprint_suite(path: Path | None = None) -> FingerprintSuite:
    """加载并校验套件；``path=None`` 时使用内置套件。"""
    resolved = path if path is not None else default_fingerprint_suite_path()
    if not resolved.exists():
        raise FingerprintSuiteNotFoundError(f"fingerprint suite not found: {resolved}")
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FingerprintSuiteValidationError(f"fingerprint suite {resolved} is unreadable: {exc}") from exc
    if not isinstance(raw, dict):
        raise FingerprintSuiteValidationError(f"fingerprint suite {resolved} must be a JSON object")
    return build_fingerprint_suite(cast(dict[str, Any], raw), source=str(resolved))


def verify_fingerprint_suite(suite: FingerprintSuite) -> str:
    """重算并校验套件内容身份，返回已验证的摘要."""
    actual = compute_suite_content_sha256(suite)
    if actual != suite.content_sha256:
        raise FingerprintSuiteIntegrityError(
            f"fingerprint suite '{suite.suite_id}' v{suite.suite_version} content hash mismatch: "
            f"recomputed {actual!r} != declared {suite.content_sha256!r}"
        )
    return actual


def probe_order(
    probes: Sequence[FingerprintProbe],
    *,
    seed: int,
    round_index: int,
) -> list[FingerprintProbe]:
    """Deterministic seeded shuffle：每轮的 probe 顺序（Task 8.1）。

    同一 ``(seed, round_index)`` 永远得到同一顺序；不同轮次得到不同顺序，
    避免"probe A 连续 16 次 → probe B 连续 16 次"把时间漂移与 probe 身份混在一起。
    """
    import random

    ordered = list(probes)
    rng = random.Random(seed + round_index)
    rng.shuffle(ordered)
    return ordered


__all__: list[str] = [
    "FingerprintSuiteError",
    "FingerprintSuiteNotFoundError",
    "FingerprintSuiteValidationError",
    "FingerprintSuiteIntegrityError",
    "DEFAULT_SUITE_FILENAME",
    "default_fingerprint_suite_path",
    "suite_content_payload",
    "compute_suite_content_sha256",
    "compute_generation_config_sha256",
    "build_fingerprint_suite",
    "load_fingerprint_suite",
    "verify_fingerprint_suite",
    "probe_order",
]
