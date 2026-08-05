"""密钥脱敏与安全控制."""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlparse, urlunparse

# 需要脱敏的请求头
SENSITIVE_HEADERS = {"authorization", "x-api-key", "api-key", "cookie", "set-cookie"}

# URL 查询参数中需要脱敏的键
SENSITIVE_QUERY_KEYS = {"token", "key", "secret", "signature", "api_key", "apikey"}

# JSON 中常见的密钥字段
SENSITIVE_JSON_KEYS = {"api_key", "apikey", "token", "secret", "password", "key"}


def redact_header_value(value: str) -> str:
    """脱敏请求头值，保留前 4 和后 4 个字符的结构."""
    if len(value) <= 8:
        return "[REDACTED]"
    return f"{value[:4]}***{value[-4:]}"


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """脱敏请求头字典."""
    result: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in SENSITIVE_HEADERS:
            result[key] = redact_header_value(value)
        else:
            result[key] = value
    return result


def redact_url(url: str) -> str:
    """脱敏 URL 中的敏感查询参数."""
    parsed = urlparse(url)
    if not parsed.query:
        return url

    params = parse_qs(parsed.query, keep_blank_values=True)
    redacted_params: dict[str, list[str]] = {}
    for key, values in params.items():
        if key.lower() in SENSITIVE_QUERY_KEYS:
            redacted_params[key] = ["[REDACTED]"]
        else:
            redacted_params[key] = values

    # 重建查询字符串
    new_query_parts = []
    for key, values in redacted_params.items():
        for v in values:
            new_query_parts.append(f"{key}={v}")

    new_query = "&".join(new_query_parts)
    new_parsed = parsed._replace(query=new_query)
    return urlunparse(new_parsed)


def redact_json_body(body: dict[str, Any] | None) -> dict[str, Any] | None:
    """脱敏 JSON 请求体中的敏感字段."""
    if body is None:
        return None
    return _redact_dict(body)


def _redact_dict(data: dict[str, Any]) -> dict[str, Any]:
    """递归脱敏字典."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if key.lower() in SENSITIVE_JSON_KEYS:
            result[key] = "[REDACTED]"
        elif isinstance(value, dict):
            result[key] = _redact_dict(value)
        elif isinstance(value, list):
            result[key] = [_redact_item(item) for item in value]
        else:
            result[key] = value
    return result


def _redact_item(item: Any) -> Any:
    """脱敏列表项."""
    if isinstance(item, dict):
        return _redact_dict(item)
    if isinstance(item, list):
        return [_redact_item(i) for i in item]
    return item


def check_api_key(env_var: str) -> str | None:
    """检查环境变量中的 API 密钥是否存在."""
    import os

    value = os.environ.get(env_var)
    if not value:
        return None
    return value


def sanitize_for_html(text: str) -> str:
    """HTML 转义，防止 XSS."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )
