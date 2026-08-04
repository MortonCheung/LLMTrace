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


def short_id(length: int = 8) -> str:
    """生成短随机 ID."""
    import secrets

    return secrets.token_hex(length // 2)
