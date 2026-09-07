"""SQLite 运行索引 —— 只保存索引 / 状态 / 摘要 / 路径，绝不保存 API Key。

真正的详细审计事实（evidence、item 结果、报告工件）继续由
``RunArtifactRepository`` 负责；这里只有一薄层 ``runs`` 表支撑 History、
Result 摘要与重启恢复（Scenario F）。

并发模型：单进程单连接 + 进程内锁，适配本地 1–3 个并发 run 的规模。
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id                TEXT PRIMARY KEY,
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL,
    target_id             TEXT NOT NULL,
    base_url_redacted     TEXT NOT NULL,
    protocol              TEXT NOT NULL,
    claimed_model         TEXT NOT NULL,

    status                TEXT NOT NULL,
    stage                 TEXT,

    reference_set_id      TEXT,
    reference_set_path    TEXT,

    suite_id              TEXT,
    suite_version         TEXT,

    capability_score      REAL,
    risk_level            TEXT,
    claimed_gap           REAL,
    confidence            TEXT,

    request_count         INTEGER NOT NULL DEFAULT 0,
    input_tokens          INTEGER NOT NULL DEFAULT 0,
    output_tokens         INTEGER NOT NULL DEFAULT 0,
    duration_ms           INTEGER NOT NULL DEFAULT 0,

    progress_completed    INTEGER NOT NULL DEFAULT 0,
    progress_total        INTEGER NOT NULL DEFAULT 0,

    artifact_path         TEXT,
    report_json_path      TEXT,
    report_html_path      TEXT,
    baseline_execution_id TEXT,

    config_json           TEXT,
    plan_json             TEXT,
    pricing_input_per_million   REAL,
    pricing_output_per_million  REAL,
    error_summary         TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC);
"""


@dataclass(frozen=True)
class RunRecord:
    """一行 run 索引记录；API Key 永不进入本结构。"""

    run_id: str
    target_id: str
    base_url_redacted: str
    protocol: str
    claimed_model: str
    status: str
    created_at: str
    updated_at: str

    stage: str | None = None
    reference_set_id: str | None = None
    reference_set_path: str | None = None
    suite_id: str | None = None
    suite_version: str | None = None
    capability_score: float | None = None
    risk_level: str | None = None
    claimed_gap: float | None = None
    confidence: str | None = None
    request_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0
    progress_completed: int = 0
    progress_total: int = 0
    artifact_path: str | None = None
    report_json_path: str | None = None
    report_html_path: str | None = None
    baseline_execution_id: str | None = None
    config_json: str | None = None
    plan_json: str | None = None
    pricing_input_per_million: float | None = None
    pricing_output_per_million: float | None = None
    error_summary: str | None = None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class RunIndex:
    """SQLite-backed 运行索引（本地单进程访问）。"""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = str(db_path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- Write ---------------------------------------------------------------

    def upsert_run(self, record: RunRecord) -> None:
        """按 run_id 插入或全量更新（本地状态机语义，非 append-only）。"""
        data = asdict(record)
        data["updated_at"] = _now_iso()
        columns = list(data.keys())
        placeholders = ",".join(":" + c for c in columns)
        update_cols = ",".join(f"{c}=excluded.{c}" for c in columns if c != "run_id")
        sql = (
            f"INSERT INTO runs ({','.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT(run_id) DO UPDATE SET {update_cols}"
        )
        with self._lock:
            self._conn.execute(sql, data)
            self._conn.commit()

    # -- Read ----------------------------------------------------------------

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return _row_to_record(row) if row is not None else None

    def list_runs(self, limit: int = 100) -> list[RunRecord]:
        """按创建时间倒序返回最近运行。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def count(self) -> int:
        with self._lock:
            (count,) = self._conn.execute("SELECT COUNT(*) FROM runs").fetchone()
        return int(count)


def _row_to_record(row: sqlite3.Row) -> RunRecord:
    names = {f.name for f in fields(RunRecord)}
    data = dict(row)
    return RunRecord(**{name: data[name] for name in names if name in data})
