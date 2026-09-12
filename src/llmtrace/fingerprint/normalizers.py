"""Choice normalizers — 第一版刻意严格（Task 6）。

只接受"整段文本恰好等于某个选项（经 NFKC + strip + casefold）"，
不做子串匹配。否则 ``"Maybe circle, but square also works"`` 会被误判成
选了 circle；无法判定时返回保留无效值，绝不"猜最接近的选项"。
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable

from llmtrace.fingerprint.models import INVALID_OUTCOME

EXACT_CHOICE_POLICY_ID = "llmtrace-exact-choice"
EXACT_CHOICE_POLICY_VERSION = "1.0.0"

#: 归一化函数签名：``(raw_text, choices) -> (outcome, valid)``。
ChoiceNormalizer = Callable[[str, tuple[str, ...]], tuple[str, bool]]


class UnsupportedNormalizationPolicyError(Exception):
    """套件声明的归一化策略不在本版本实现范围内（fail closed）."""

    error_code = "UNSUPPORTED_NORMALIZATION_POLICY"


def normalize_exact_choice(
    text: str,
    choices: tuple[str, ...],
) -> tuple[str, bool]:
    """严格整段匹配：命中返回 ``(canonical_choice, True)``，否则 ``(INVALID, False)``."""
    value = unicodedata.normalize("NFKC", text)
    value = value.strip().casefold()

    mapping = {unicodedata.normalize("NFKC", choice).strip().casefold(): choice for choice in choices}

    if value in mapping:
        return mapping[value], True

    return INVALID_OUTCOME, False


#: 已实现的归一化策略身份集合；新增策略必须显式登记，未知策略一律 fail closed。
_SUPPORTED_POLICIES: frozenset[tuple[str, str]] = frozenset({(EXACT_CHOICE_POLICY_ID, EXACT_CHOICE_POLICY_VERSION)})


def resolve_normalizer(policy_id: str, policy_version: str) -> ChoiceNormalizer:
    """按策略身份解析归一化函数；未知策略 fail closed."""
    if (policy_id, policy_version) in _SUPPORTED_POLICIES:
        return normalize_exact_choice
    raise UnsupportedNormalizationPolicyError(
        f"unsupported fingerprint normalization policy {policy_id!r} v{policy_version!r}; "
        f"supported: {sorted(_SUPPORTED_POLICIES)}"
    )
