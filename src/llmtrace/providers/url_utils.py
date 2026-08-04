"""URL 拼接工具."""

from __future__ import annotations


def join_url(base: str, path: str) -> str:
    """安全拼接 URL 路径，避免重复 /v1 前缀."""
    base = base.rstrip("/")
    path = path.lstrip("/")

    # 检查路径是否已有 /v1 前缀，且 base 也以 /v1 结尾
    base_parts = base.split("/")
    path_parts = path.split("/")

    # 如果 base 以 /v1 结尾，且 path 以 v1/ 开头，去除重复
    if base_parts[-1] == "v1" and path_parts[0] == "v1":
        return "/".join(base_parts + path_parts[1:])

    return f"{base}/{path}"
