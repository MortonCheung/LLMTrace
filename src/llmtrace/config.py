"""审计配置数据模型."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class Protocol(StrEnum):
    """支持的协议类型."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class AuthStyle(StrEnum):
    """鉴权方式."""

    AUTO = "auto"
    BEARER = "bearer"
    X_API_KEY = "x-api-key"
    BOTH = "both"


class AuditConfig(BaseModel):
    """审计配置."""

    protocol: Protocol
    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key_env: str = Field(min_length=1)
    auth_style: AuthStyle = AuthStyle.AUTO
    repeat_count: int = Field(default=3, ge=1, le=10)
    timeout: float = Field(default=30.0, ge=1.0, le=300.0)
    max_response_bytes: int = Field(default=64 * 1024, ge=1024, le=10 * 1024 * 1024)
    max_output_tokens: int = Field(default=64, ge=1, le=4096)
    check_streaming: bool = True
    output_dir: Path = Field(default=Path("reports"))
    test_suite_version: str = "1.0"

    @model_validator(mode="after")
    def _validate_url(self) -> AuditConfig:
        url = self.base_url.rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        self.base_url = url
        return self
