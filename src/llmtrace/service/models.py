"""RunService 输入与估算数据模型（薄 Service 层）。

原则（§八 / §二十二）：API Key 永不进入任何持久化结构。Web 提交的
key 只存在于 :class:`RunService` 的内存 dict，绝不写入 SQLite / JSON。
``config_json`` 落盘时使用经过 ``redact_url`` 处理的 base_url，因此重启后
可安全展示历史（Scenario F），但缺少完整凭据的 run 不能再次 start。

本模块刻意使用 dataclass 而非 pydantic：校验语义归 Web API 层
（FastAPI 请求体模型），Service 层只接收已经过形状校验的输入。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CreateRunInput:
    """一次 Web 创建的完整输入（api_key 仅存内存）。"""

    base_url: str
    model: str
    api_key: str
    protocol: str = "openai"
    auth_style: str = "auto"
    repeat: int = 3
    timeout: float = 30.0
    check_streaming: bool = True
    reference_set_path: str | None = None


@dataclass(frozen=True)
class RunEstimate:
    """create → estimate 的脱机估算结果（不发送任何 HTTP）。"""

    plan_id: str
    target_id: str
    candidate_model_id: str
    suite_id: str
    suite_version: str
    protocol_probe_requests: int
    benchmark_requests: int
    planned_requests: int
    maximum_requests: int
    maximum_output_token_ceiling: int
    estimated_cost: float | None
    requires_secure_code_sandbox: bool
    reference_calibration: bool
    reference_set_id: str | None
