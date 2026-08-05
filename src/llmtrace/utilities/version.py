"""版本信息."""

from __future__ import annotations

import platform
import sys

from llmtrace import __version__


def get_llmtrace_version() -> str:
    """获取 LLMTrace 版本."""
    return __version__


def get_python_version() -> str:
    """获取 Python 版本."""
    return sys.version.split()[0]


def get_platform() -> str:
    """获取平台信息."""
    return f"{platform.system()} {platform.release()}"
