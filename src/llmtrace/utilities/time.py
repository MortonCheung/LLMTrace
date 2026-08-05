"""时间工具."""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    """当前 UTC 时间."""
    return datetime.now(tz=UTC)


def format_iso(dt: datetime) -> str:
    """格式化为 ISO 8601 字符串."""
    return dt.isoformat()


def format_file_time(dt: datetime) -> str:
    """格式化为文件名安全的时间字符串."""
    return dt.strftime("%Y%m%d_%H%M%S")
