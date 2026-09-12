"""Measurement resolution 声明（Task 33）.

Quick Suite 是低成本的 screening 测量：它能回答"这个 API 的行为看起来像不像它
自称的模型"，不能回答"身份已被证明"。本轮**不改** Calibration formula，也不定义
任何 ± 容差数字 —— 除非以后通过完整 benchmark meta-evaluation 得出（Task 33）。

这里只提供报告要用的、可机器读取的声明文本；渲染由现有报告框架负责。
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

#: 报告里该声明所在的小节名。
MEASUREMENT_RESOLUTION_LABEL = "Measurement Resolution"

#: 报告里该声明的取值（Task 33 指定的大写形式）。
QUICK_SUITE_MEASUREMENT_RESOLUTION = "SCREENING"

#: Quick Suite 的低分辨率说明（Task 33 原文）.
QUICK_SUITE_RESOLUTION_DISCLAIMER = (
    "Quick Suite is a low-cost screening measurement. Small score differences "
    "should not be interpreted as model-identity evidence without independent validation."
)


class MeasurementResolution(StrEnum):
    """测量分辨率等级；当前只有 Quick Suite 的 screening 一级."""

    SCREENING = QUICK_SUITE_MEASUREMENT_RESOLUTION


class MeasurementResolutionNote(BaseModel):
    """一条可写入 JSON / HTML 的分辨率声明（不含任何容差数字）."""

    label: str = Field(default=MEASUREMENT_RESOLUTION_LABEL, min_length=1)
    resolution: MeasurementResolution
    disclaimer: str = Field(..., min_length=1)

    model_config = {"frozen": True, "extra": "forbid"}


def quick_suite_resolution_note() -> MeasurementResolutionNote:
    """Quick Suite 使用的分辨率声明."""
    return MeasurementResolutionNote(
        resolution=MeasurementResolution.SCREENING,
        disclaimer=QUICK_SUITE_RESOLUTION_DISCLAIMER,
    )


__all__: list[str] = [
    "MEASUREMENT_RESOLUTION_LABEL",
    "QUICK_SUITE_MEASUREMENT_RESOLUTION",
    "QUICK_SUITE_RESOLUTION_DISCLAIMER",
    "MeasurementResolution",
    "MeasurementResolutionNote",
    "quick_suite_resolution_note",
]
