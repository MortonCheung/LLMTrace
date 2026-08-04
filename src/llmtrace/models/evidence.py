"""证据数据模型."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HTTPEvidence(BaseModel):
    """单次 HTTP 请求-响应证据."""

    # 请求信息
    request_method: str
    request_url_redacted: str
    request_path: str
    request_headers_redacted: dict[str, str]
    request_body_redacted: dict[str, Any] | None = None
    request_time: datetime | None = None

    # 响应信息
    response_time: datetime | None = None
    http_status: int | None = None
    response_headers: dict[str, str] = Field(default_factory=dict)
    response_body_summary: str = ""
    response_body_sha256: str = ""
    response_truncated: bool = False
    response_body_size: int = 0

    # 性能
    first_token_latency_ms: float | None = None
    total_latency_ms: float | None = None

    # 协议字段
    request_id: str | None = None
    response_id: str | None = None
    request_model: str | None = None
    response_model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    response_text: str = ""

    # 异常
    exception_type: str | None = None
    exception_message: str | None = None

    @property
    def success(self) -> bool:
        """请求是否成功."""
        return self.http_status is not None and 200 <= self.http_status < 300 and self.exception_type is None
