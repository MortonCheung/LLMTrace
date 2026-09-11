"""本地应用数据目录（local-first，v0.5 Usable MVP）。

默认根目录 ``~/.llmtrace``（可被环境变量 ``LLMTRACE_HOME`` 或 CLI
``--data-dir`` 覆盖）：

    ~/.llmtrace/
      llmtrace.db          # SQLite 运行索引（绝不含 API Key）
      runs/<run_id>/...    # RunArtifactRepository 工件（含 report.json/html）
      references/          # reference sets / snapshots（与 CLI capture 兼容）
      fingerprints/        # v0.6 身份证据：snapshots / sets / policies

数据目录本身不保存 API Key；Key 只在进程内存中存在。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_DATA_ROOT = "LLMTRACE_HOME"


def default_data_root() -> Path:
    """默认应用数据目录：``$LLMTRACE_HOME`` 或 ``~/.llmtrace``。"""
    env = os.environ.get(ENV_DATA_ROOT)
    if env:
        return Path(env).expanduser()
    return Path.home() / ".llmtrace"


@dataclass(frozen=True)
class AppLayout:
    """从 data root 派生的固定布局。"""

    root: Path
    db_path: Path
    references_dir: Path
    reference_sets_dir: Path
    reference_snapshots_dir: Path
    fingerprints_dir: Path
    fingerprint_snapshots_dir: Path
    fingerprint_sets_dir: Path
    fingerprint_policies_dir: Path

    @classmethod
    def resolve(cls, data_root: Path | None = None) -> AppLayout:
        root = data_root.expanduser().resolve() if data_root is not None else default_data_root().resolve()
        fingerprints_dir = root / "fingerprints"
        return cls(
            root=root,
            db_path=root / "llmtrace.db",
            references_dir=root / "references",
            reference_sets_dir=root / "references" / "sets",
            reference_snapshots_dir=root / "references" / "snapshots",
            fingerprints_dir=fingerprints_dir,
            fingerprint_snapshots_dir=fingerprints_dir / "snapshots",
            fingerprint_sets_dir=fingerprints_dir / "sets",
            fingerprint_policies_dir=fingerprints_dir / "policies",
        )


def ensure_app_layout(data_root: Path | None = None) -> AppLayout:
    """创建并返回应用布局（幂等）。"""
    layout = AppLayout.resolve(data_root)
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.references_dir.mkdir(parents=True, exist_ok=True)
    layout.fingerprints_dir.mkdir(parents=True, exist_ok=True)
    return layout
