"""薄 Service Layer 的 :class:`RunService`（§十三）。

职责（create / estimate / start / progress / cancel / result / history），
只封装 :class:`UnifiedAuditRunner`，本身不重新实现 audit（§十三）。

执行模型：单进程 asyncio + ``run_id -> asyncio.Task`` 内存表（§十四），
不引入 Celery / Redis。进程重启导致进行中的 run 丢失是首版允许的边界，
历史结果始终在磁盘 + SQLite（Scenario F）。

安全（§八 / §二十二 / §六十）：API Key 只存在于本进程内存；任何持久化
（SQLite config_json、artifacts、事件）都不含 key。config_json 使用
redacted base_url 落盘，因此历史与 estimate 可安全恢复，但 start 永远
需要当前进程内存里的完整配置与 key。
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llmtrace.adapters.code_execution import (
    CodeExecutionBackend,
    SandboxUnavailableError,
)
from llmtrace.appdir import AppLayout
from llmtrace.config import AuditConfig, AuthStyle, Protocol
from llmtrace.execution.artifacts import RunArtifactRepository
from llmtrace.execution.planner import (
    build_unified_execution_plan,
    derive_target_id,
)
from llmtrace.execution.progress import (
    STAGE_CANCELLED,
    STAGE_DONE,
    STAGE_FAILED,
    CancellationToken,
    ProgressEvent,
    ProgressSink,
)
from llmtrace.execution.runner import UnifiedAuditRunner
from llmtrace.security.redaction import redact_url
from llmtrace.service.models import CreateRunInput, RunEstimate
from llmtrace.storage.sqlite import RunIndex, RunRecord

# ---------------------------------------------------------------------------
# 错误
# ---------------------------------------------------------------------------


class RunServiceError(Exception):
    """Service 层错误基类（Web API 层统一映射为 4xx/5xx）。"""


class RunNotFoundError(RunServiceError):
    error_code = "RUN_NOT_FOUND"


class RunStateConflictError(RunServiceError):
    error_code = "RUN_STATE_CONFLICT"


class RunCreateError(RunServiceError):
    error_code = "RUN_CREATE_INVALID"


class RunEstimateError(RunServiceError):
    error_code = "RUN_ESTIMATE_UNAVAILABLE"


class RunKeyUnavailableError(RunServiceError):
    error_code = "RUN_KEY_UNAVAILABLE"


class RunPreflightError(RunServiceError):
    """start 前的本地预检失败（如 Docker sandbox 不可用）。"""

    error_code = "RUN_PREFLIGHT_FAILED"


# 阶段 tag —— service 自己不下发中间事件，只在 run 结束时补一条终态事件，
# 供 SSE 端收到后跳转 result 页面。
TERMINAL_EVENT = "status"

# Web 会话通过固定 env 名通过协议预检的 key 存在性检查（与 CLI env 机制同构）。
# 该变量的值只是非空占位符，绝不含真实 key —— 真实 key 只存在于
# UnifiedAuditRunner 的 provider 内存中。
WEB_KEY_ENV = "LLMTRACE_WEB_SESSION_KEY"
_WEB_KEY_PLACEHOLDER = "provided-by-web-ui"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _redacted_record_config(config: AuditConfig) -> str:
    """落盘的 config 使用 redacted base_url —— 绝不写入完整 URL 凭据。"""
    redacted = config.model_copy(update={"base_url": redact_url(config.base_url)})
    return redacted.model_dump_json()


# ---------------------------------------------------------------------------
# 内存运行态
# ---------------------------------------------------------------------------


@dataclass
class _LiveRun:
    """单次 run 的内存状态（进程存活期间保留，便于 result/SSE 重放）。"""

    run_id: str
    cancel_token: CancellationToken = field(default_factory=CancellationToken)
    task: asyncio.Task[Any] | None = None
    # 事件重放缓冲（sink 同步 append，消费在同一 event loop 进行）。
    events: deque[ProgressEvent] = field(default_factory=lambda: deque(maxlen=2000))
    wake: asyncio.Event | None = None
    terminal_status: str | None = None
    error_summary: str | None = None


# ---------------------------------------------------------------------------
# RunService
# ---------------------------------------------------------------------------


class RunService:
    """面向 Web 的薄运行服务；一个进程共享一个实例（FastAPI app.state）。"""

    def __init__(
        self,
        layout: AppLayout,
        *,
        code_backend: CodeExecutionBackend | None = None,
    ) -> None:
        self._layout = layout
        self._index = RunIndex(layout.db_path)
        # None => runner 使用生产默认（Docker，失败则拒绝）。Web 的 --demo
        # 模式会显式注入受信任的测试后端。
        self._code_backend = code_backend
        self._runs: dict[str, _LiveRun] = {}
        self._configs: dict[str, AuditConfig] = {}
        self._keys: dict[str, str] = {}
        self._lock = threading.RLock()
        self._closed = False

    # -- 生命周期 ---------------------------------------------------------

    def close(self) -> None:
        """关闭 SQLite 连接（Web lifespan shutdown 调用）。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._index.close()

    # -- create / estimate -------------------------------------------------

    def create(self, run_input: CreateRunInput) -> RunRecord:
        """校验输入、派生 target_id、建立 run 索引（持久化，绝不含 key）。"""
        self._ensure_open()
        api_key = (run_input.api_key or "").strip()
        if not api_key:
            raise RunCreateError("api key is required")
        try:
            config = AuditConfig(
                protocol=Protocol(run_input.protocol),
                base_url=run_input.base_url,
                model=run_input.model,
                # 占位 env 名：协议预检的 key 存在性检查走 CLI 同款 env 机制，
                # 而真实 key 只由 provider 在内存持有，永不落盘。
                api_key_env=WEB_KEY_ENV,
                auth_style=AuthStyle(run_input.auth_style),
                repeat_count=run_input.repeat,
                timeout=run_input.timeout,
                check_streaming=run_input.check_streaming,
                output_dir=self._layout.root,
            )
        except (ValueError, KeyError) as exc:
            raise RunCreateError(str(exc)) from exc

        target_id = derive_target_id(config.protocol.value, config.base_url)
        reference_set_path = run_input.reference_set_path
        if reference_set_path is not None and not Path(reference_set_path).is_file():
            raise RunCreateError(f"reference set not found: {reference_set_path}")

        run_id = uuid.uuid4().hex
        now = _now_iso()
        record = RunRecord(
            run_id=run_id,
            target_id=target_id,
            base_url_redacted=redact_url(config.base_url),
            protocol=str(config.protocol.value),
            claimed_model=config.model,
            status="PENDING",
            created_at=now,
            updated_at=now,
            reference_set_path=reference_set_path,
            config_json=_redacted_record_config(config),
        )
        with self._lock:
            self._configs[run_id] = config
            self._keys[run_id] = api_key
            self._index.upsert_run(record)
        return record

    def estimate(self, run_id: str) -> RunEstimate:
        """脱机估算：不发送 HTTP、不创建 provider、不运行任何代码（§十三）。"""
        record = self._require_record(run_id)
        config = self._resolve_config(run_id, record)

        # Reference calibration 上下文（脱机、只读）：
        reference_set_id: str | None = None
        reference_set_version: str | None = None
        reference_set_sha: str | None = None
        calibration_policy_id: str | None = None
        calibration_policy_version: str | None = None
        if record.reference_set_path is not None:
            try:
                from llmtrace.reference.validation import validate_reference_set_for_calibration

                context = validate_reference_set_for_calibration(
                    set_path=Path(record.reference_set_path),
                    artifact_repository=RunArtifactRepository(self._layout.root),
                )
                reference_set_id = context.reference_set.reference_set_id
                reference_set_version = context.reference_set.reference_set_version
                reference_set_sha = context.reference_set.content_sha256
                calibration_policy_id = context.calibration_policy.policy_id
                calibration_policy_version = context.calibration_policy.policy_version
            except Exception as exc:  # noqa: BLE001 —— 映射为可读的估算错误
                raise RunEstimateError(str(exc)) from exc

        try:
            plan = build_unified_execution_plan(
                config,
                target_id=record.target_id,
                reference_set_id=reference_set_id,
                reference_set_version=reference_set_version,
                reference_set_content_sha256=reference_set_sha,
                calibration_policy_id=calibration_policy_id,
                calibration_policy_version=calibration_policy_version,
            )
        except Exception as exc:  # noqa: BLE001
            raise RunEstimateError(str(exc)) from exc

        return RunEstimate(
            plan_id=plan.plan_id,
            target_id=plan.target_id,
            candidate_model_id=plan.candidate_model_id,
            suite_id=plan.suite_id,
            suite_version=plan.suite_version,
            protocol_probe_requests=plan.protocol_probe_requests,
            benchmark_requests=plan.benchmark_requests,
            planned_requests=plan.planned_requests,
            maximum_requests=plan.maximum_requests,
            maximum_output_token_ceiling=plan.maximum_output_token_ceiling,
            estimated_cost=plan.estimated_cost,
            requires_secure_code_sandbox=plan.requires_secure_code_sandbox,
            reference_calibration=reference_set_id is not None,
            reference_set_id=reference_set_id,
        )

    # -- start / cancel ----------------------------------------------------

    def start(self, run_id: str) -> None:
        """以 asyncio.Task 在后台启动一次完整 audit；立即返回。

        需当前进程内存中的 API Key；重启后的历史 run 没有 key，会以
        :class:`RunKeyUnavailableError` 拒绝（不会要求重新输入 key 才能 start）。
        """
        self._ensure_open()
        record = self._require_record(run_id)
        if record.status != "PENDING":
            raise RunStateConflictError(f"run {run_id} is already {record.status}")

        config = self._resolve_config(run_id, record)
        api_key = self._keys.get(run_id)
        if not api_key:
            raise RunKeyUnavailableError("api key is not available in this server process; create a new run")

        live = self._runs.get(run_id)
        if live is not None and live.task is not None and not live.task.done():
            raise RunStateConflictError(f"run {run_id} is already running")

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:  # pragma: no cover —— Web/async 环境保证
            raise RunStateConflictError("start() must run inside an asyncio loop") from exc

        if live is None:
            live = _LiveRun(run_id=run_id)
            self._runs[run_id] = live
        live.wake = asyncio.Event()
        live.terminal_status = None
        live.error_summary = None

        self._upsert_status(run_id, "RUNNING", stage=None)

        # 协议预检通过 env 检查 key 是否可用；值为非空占位符即可（真实 key
        # 只经 runner 的 provider 内存传递，绝不写进环境变量或磁盘）。
        os.environ[WEB_KEY_ENV] = _WEB_KEY_PLACEHOLDER

        try:
            runner = UnifiedAuditRunner(
                config,
                api_key=api_key,
                target_id=record.target_id,
                repository=RunArtifactRepository(self._layout.root),
                code_backend=self._code_backend,
                compare_latest=True,
                baseline_snapshot_path=None,
                reference_snapshot_path=None,
                reference_set_path=Path(record.reference_set_path) if record.reference_set_path else None,
                progress_sink=self._make_sink(live),
                cancel_token=live.cancel_token,
            )
        except SandboxUnavailableError as exc:
            self._upsert_status(run_id, "FAILED", stage=STAGE_FAILED, error_summary=str(exc))
            self._append_terminal(live, "failed", str(exc))
            raise RunPreflightError(str(exc)) from exc

        live.task = loop.create_task(self._run_and_finalize(run_id, runner, live))

    def cancel(self, run_id: str) -> dict[str, str]:
        """置位 cooperative cancellation（§二十）；幂等。"""
        record = self._require_record(run_id)
        live = self._runs.get(run_id)
        if record.status in ("PENDING",) or (live is not None and live.task is not None and not live.task.done()):
            if live is not None:
                live.cancel_token.cancel()
                return {"run_id": run_id, "status": "CANCELLING"}
            return {"run_id": run_id, "status": "IDLE"}
        return {"run_id": run_id, "status": record.status}

    # -- 查询 --------------------------------------------------------------

    def progress(self, run_id: str) -> dict[str, Any]:
        """进度快照：最新索引状态 + 事件重放缓冲（供 SSE / 轮询共用）。"""
        record = self._require_record(run_id)
        live = self._runs.get(run_id)
        events: list[dict[str, Any]] = []
        terminal_status = None
        error_summary = record.error_summary
        if live is not None:
            events = [e.to_dict() for e in live.events]
            terminal_status = live.terminal_status
            error_summary = error_summary or live.error_summary
        return {
            "run_id": run_id,
            "status": record.status,
            "stage": record.stage,
            "events": events,
            "terminal": terminal_status is not None,
            "terminal_status": terminal_status,
            "error_summary": error_summary,
        }

    def result(self, run_id: str) -> dict[str, Any]:
        """终态结果视图：run 索引 + report.json + 报告文件路径。

        report.json 已在写入边界做过 secret scrub（项目既有保证），直接读取。
        """
        record = self._require_record(run_id)
        payload: dict[str, Any] = {"run": _record_view(record), "report": None}
        if record.report_json_path:
            report_path = Path(record.report_json_path)
            if report_path.is_file():
                try:
                    payload["report"] = json.loads(report_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    payload["report"] = None
        return payload

    def view(self, run_id: str) -> dict[str, Any]:
        """公开单条 run 记录视图（不含任何 secret / 内部 config）。"""
        return _record_view(self._require_record(run_id))

    def history(self, limit: int = 100) -> list[dict[str, Any]]:
        """按创建时间倒序返回历史 run（不含任何 secret）。"""
        self._ensure_open()
        return [_record_view(r) for r in self._index.list_runs(limit=limit)]

    # -- 内部：终态落盘 ------------------------------------------------------

    async def _run_and_finalize(self, run_id: str, runner: UnifiedAuditRunner, live: _LiveRun) -> None:
        """后台任务：完整 audit → 终态 record（原子化 upsert）→ 终态事件。"""
        started = datetime.now(UTC)
        try:
            result = await runner.run()
        except Exception as exc:  # noqa: BLE001 —— runner 已 emit failed，这里落盘
            error = self._scrub_text(run_id, str(exc))
            self._upsert_status(
                run_id,
                "FAILED",
                stage=STAGE_FAILED,
                error_summary=error,
            )
            live.error_summary = error
            self._append_terminal(live, "failed", error)
            return

        finished = datetime.now(UTC)
        duration_ms = int((finished - started).total_seconds() * 1000)
        status = str(result.status.value)  # COMPLETED / *_WARNINGS / PARTIAL / CANCELLED
        stage = STAGE_DONE
        if status == "CANCELLED":
            stage = STAGE_CANCELLED
        elif status == "FAILED":
            stage = STAGE_FAILED

        record = self._require_record(run_id)
        update: dict[str, Any] = {
            "run_id": run_id,
            "target_id": record.target_id,
            "base_url_redacted": record.base_url_redacted,
            "protocol": record.protocol,
            "claimed_model": record.claimed_model,
            "created_at": record.created_at,
            "updated_at": _now_iso(),
            "status": status,
            "stage": stage,
            "reference_set_id": record.reference_set_id,
            "reference_set_path": record.reference_set_path,
            "config_json": record.config_json,
        }

        profile = result.capability_profile
        if profile is not None:
            update["capability_score"] = (
                profile.calibrated_total_score
                if profile.calibrated_total_score is not None
                else round(profile.provisional_raw_index * 100.0, 1)
            )
        if result.claimed_model_gap is not None:
            update["claimed_gap"] = round(result.claimed_model_gap.total_delta, 2)

        update["suite_id"] = result.plan.suite_id
        update["suite_version"] = result.plan.suite_version
        update["duration_ms"] = duration_ms

        requests = runner.request_budget.consumed_requests if runner.request_budget is not None else 0
        update["request_count"] = requests

        # Confidence v1（实验性；fail-closed —— 无测量返回 Unavailable）。
        try:
            from llmtrace.analysis.confidence import ConfidencePolicy

            assessment = ConfidencePolicy.create_v1().assess(
                measurement=result.measurement_summary,
                capability_profile=profile,
            )
            update["confidence"] = str(assessment.level.value)
        except Exception:  # noqa: BLE001 —— 元数据增强失败不影响 run 本身
            pass

        execution_id = result.execution_id
        run_dir = self._layout.root / "runs" / execution_id
        update["artifact_path"] = str(run_dir)
        update["report_json_path"] = str(run_dir / "report.json")
        update["report_html_path"] = str(run_dir / "report.html")

        # baseline_execution_id 与 actual_requests 的权威来源是 manifest。
        try:
            manifest = RunArtifactRepository(self._layout.root).load_manifest(execution_id)
            update["baseline_execution_id"] = manifest.baseline_execution_id
            update["request_count"] = manifest.actual_requests
        except Exception:  # noqa: BLE001 —— manifest 缺失时保持基本字段
            pass

        final_record = _record_from_update(record, update)
        with self._lock:
            self._index.upsert_run(final_record)
            # 终态后立即清理进程内存里的凭据（key / 完整 config 不再需要）。
            self._keys.pop(run_id, None)
            self._configs.pop(run_id, None)
        live.terminal_status = status.lower()
        self._append_terminal(live, status.lower(), None)

    # -- 内部：事件与工具 ----------------------------------------------------

    def _make_sink(self, live: _LiveRun) -> ProgressSink:
        def sink(event: ProgressEvent) -> None:
            # 同步回调：runner 运行于本进程 event loop，append + set 线程安全。
            live.events.append(event)
            if live.wake is not None:
                live.wake.set()

        return sink

    def _append_terminal(self, live: _LiveRun, status: str, error: str | None) -> None:
        """向事件缓冲补一条终态事件，SSE 客户端据此跳转 result 页。"""
        event = ProgressEvent(
            type=TERMINAL_EVENT,
            stage=status,
            message=error or f"run {status}",
            extra={"run_status": status},
        )
        live.events.append(event)
        if live.wake is not None:
            live.wake.set()

    def _scrub_text(self, run_id: str, text: str) -> str:
        from llmtrace.security.redaction import SecretScrubber

        key = self._keys.get(run_id)
        secrets: list[str] = [key] if key else []
        return SecretScrubber([s for s in secrets if s]).scrub_text(text)

    def _upsert_status(
        self,
        run_id: str,
        status: str,
        *,
        stage: str | None,
        error_summary: str | None = None,
    ) -> None:
        record = self._require_record(run_id)
        update = {
            "run_id": run_id,
            "target_id": record.target_id,
            "base_url_redacted": record.base_url_redacted,
            "protocol": record.protocol,
            "claimed_model": record.claimed_model,
            "created_at": record.created_at,
            "updated_at": _now_iso(),
            "status": status,
            "stage": stage,
            "reference_set_id": record.reference_set_id,
            "reference_set_path": record.reference_set_path,
            "config_json": record.config_json,
        }
        if error_summary is not None:
            update["error_summary"] = error_summary
        with self._lock:
            self._index.upsert_run(_record_from_update(record, update))

    def _resolve_config(self, run_id: str, record: RunRecord) -> AuditConfig:
        config = self._configs.get(run_id)
        if config is not None:
            return config
        # 重启恢复路径：config_json 只含 redacted base_url（无凭据），
        # 足够 estimate/history；不能 start。
        if record.config_json:
            try:
                return AuditConfig.model_validate_json(record.config_json)
            except Exception as exc:  # noqa: BLE001
                raise RunEstimateError(f"stored config invalid: {exc}") from exc
        raise RunKeyUnavailableError(f"run {run_id} has no usable configuration in memory")

    def _require_record(self, run_id: str) -> RunRecord:
        record = self._index.get_run(run_id)
        if record is None:
            raise RunNotFoundError(f"run not found: {run_id}")
        return record

    def _ensure_open(self) -> None:
        if self._closed:
            raise RunStateConflictError("service is closed")


def _record_view(record: RunRecord) -> dict[str, Any]:
    """History/结果页的公开视图 —— 明确排除 config_json/plan_json 等内部字段。"""
    return {
        "run_id": record.run_id,
        "target_id": record.target_id,
        "base_url_redacted": record.base_url_redacted,
        "protocol": record.protocol,
        "claimed_model": record.claimed_model,
        "status": record.status,
        "stage": record.stage,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "reference_set_id": record.reference_set_id,
        "suite_id": record.suite_id,
        "suite_version": record.suite_version,
        "capability_score": record.capability_score,
        "risk_level": record.risk_level,
        "claimed_gap": record.claimed_gap,
        "confidence": record.confidence,
        "request_count": record.request_count,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "duration_ms": record.duration_ms,
        "artifact_path": record.artifact_path,
        "report_json_path": record.report_json_path,
        "report_html_path": record.report_html_path,
        "baseline_execution_id": record.baseline_execution_id,
        "error_summary": record.error_summary,
    }


def _record_from_update(record: RunRecord, update: dict[str, Any]) -> RunRecord:
    """按白名单从原 record + 更新字段构造新 record（防缺省字段漂移）。"""
    values = {
        field_name: getattr(record, field_name)
        for field_name in (
            "stage",
            "reference_set_id",
            "reference_set_path",
            "suite_id",
            "suite_version",
            "capability_score",
            "risk_level",
            "claimed_gap",
            "confidence",
            "request_count",
            "input_tokens",
            "output_tokens",
            "duration_ms",
            "progress_completed",
            "progress_total",
            "artifact_path",
            "report_json_path",
            "report_html_path",
            "baseline_execution_id",
            "plan_json",
            "pricing_input_per_million",
            "pricing_output_per_million",
            "error_summary",
        )
    }
    values.update(update)
    return RunRecord(**values)
