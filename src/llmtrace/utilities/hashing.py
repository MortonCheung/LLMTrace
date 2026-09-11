"""哈希工具."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_hash(data: str | bytes) -> str:
    """计算 SHA-256 哈希."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def stable_json_hash(obj: dict[str, Any] | list[Any]) -> str:
    """计算 JSON 对象的稳定哈希."""
    canonical = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_hash(canonical)


def canonical_json(obj: Any) -> str:
    """仓库级 canonical JSON：排序键 + 紧凑分隔符 + ASCII 转义.

    与 reference/suite/behavior 的既有 canonical 模式一致（``ensure_ascii=True``），
    用于内容身份哈希；``stable_json_hash`` 的 ``ensure_ascii=False`` 语义不同，
    两者不可互换。

    调用方对 datetime 字段应先经 ``model_dump(mode="json")`` 等转为 UTC ISO-8601
    字符串，保证跨时区表示字节一致。
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_json_hash(obj: Any) -> str:
    """canonical JSON 文本的 SHA-256（内容身份）."""
    return sha256_hash(canonical_json(obj))


def short_id(length: int = 8) -> str:
    """生成短随机 ID."""
    import secrets

    return secrets.token_hex(length // 2)
