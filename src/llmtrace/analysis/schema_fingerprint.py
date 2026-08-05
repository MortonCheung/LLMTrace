"""响应结构指纹."""

from __future__ import annotations

import json
from typing import Any

from llmtrace.utilities.hashing import sha256_hash


def generate_schema_fingerprint(response_body: str) -> str | None:
    """从响应 JSON 生成结构指纹.

    指纹基于 JSON 键路径和字段类型，不包含随机 ID 或文本内容。
    """
    try:
        data = json.loads(response_body)
    except (json.JSONDecodeError, TypeError):
        return None

    paths = _extract_paths(data, "")
    sorted_paths = sorted(paths)
    canonical = json.dumps(sorted_paths, sort_keys=True)
    return sha256_hash(canonical)


def _extract_paths(obj: Any, prefix: str) -> list[str]:
    """递归提取 JSON 键路径和类型."""
    paths: list[str] = []
    if isinstance(obj, dict):
        for key, value in sorted(obj.items()):
            new_prefix = f"{prefix}.{key}" if prefix else key
            val_type = _type_name(value)
            paths.append(f"{new_prefix}:{val_type}")
            if isinstance(value, (dict, list)):
                paths.extend(_extract_paths(value, new_prefix))
    elif isinstance(obj, list) and obj:
        val_type = _type_name(obj[0])
        paths.append(f"{prefix}[]:{val_type}")
        if isinstance(obj[0], (dict, list)):
            paths.extend(_extract_paths(obj[0], f"{prefix}[]"))
    return paths


def _type_name(value: Any) -> str:
    """获取值的类型名称."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return "unknown"
