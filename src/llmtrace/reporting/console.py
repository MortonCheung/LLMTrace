"""终端控制台输出."""

from __future__ import annotations

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from llmtrace.analysis.confidence import ConfidencePolicy
from llmtrace.analysis.routing import assess_routing_stability
from llmtrace.execution.progress import (
    STAGE_BENCHMARK,
    STAGE_CALIBRATION,
    STAGE_CANCELLED,
    STAGE_COMPARISON,
    STAGE_DONE,
    STAGE_FAILED,
    STAGE_PREFLIGHT,
    STAGE_PROTOCOL,
    STAGE_REPORTING,
    STAGE_SCORING,
)
from llmtrace.fingerprint.models import FingerprintMatchStatus
from llmtrace.models.audit import AuditResult, RiskLevel
from llmtrace.security.redaction import redact_url

_console = Console()

#: "Top Behavioral Matches" 控制台预览条数；纯展示截断，不是阈值、不是 Top-K 结论。
_TOP_MATCH_DISPLAY_LIMIT = 3

#: matcher status → 控制台 "Claim Consistency" 文案（Task 42 / Task 43 / Task 20）.
#:
#: ``RANKED_ONLY`` 的含义是"只有排序、没有 validated policy 支撑的判定"，在控制台上
#: 就是 INCONCLUSIVE；它绝不能被写成任何形式的身份结论（Rule 2 / Rule 3 / Task 48）。
_CLAIM_CONSISTENCY_LABELS = {
    FingerprintMatchStatus.CONSISTENT_WITH_CLAIM: "BEHAVIOR CONSISTENT WITH CLAIM",
    FingerprintMatchStatus.INCONSISTENT_WITH_CLAIM: "BEHAVIOR INCONSISTENT WITH CLAIM",
    FingerprintMatchStatus.RANKED_ONLY: "INCONCLUSIVE",
    FingerprintMatchStatus.INCONCLUSIVE: "INCONCLUSIVE",
}


def _claim_consistency_label(status: FingerprintMatchStatus) -> str:
    """把 matcher status 映射为控制台文案；未验证 policy 一律 INCONCLUSIVE."""
    return _CLAIM_CONSISTENCY_LABELS[status]


# 阶段 token → 中文标签（TTY Live 与 非 TTY 日志共用）。
_STAGE_LABELS = {
    STAGE_PREFLIGHT: "预检",
    STAGE_PROTOCOL: "协议审计",
    STAGE_BENCHMARK: "能力基准",
    STAGE_SCORING: "能力评分",
    STAGE_CALIBRATION: "参考校准",
    STAGE_COMPARISON: "历史对比",
    STAGE_REPORTING: "生成报告",
    STAGE_DONE: "完成",
    STAGE_FAILED: "失败",
    STAGE_CANCELLED: "已取消",
}


def print_audit_summary(result: AuditResult) -> None:
    """打印审计摘要."""
    _console.print()
    _console.print(Panel.fit("LLMTrace 审计完成", style="bold blue"))

    config = result.config
    table = Table(title="审计摘要")
    table.add_column("项目", style="cyan")
    table.add_column("值", style="white")

    table.add_row("Endpoint", redact_url(config.base_url))
    table.add_row("协议", config.protocol.value)
    table.add_row("声称模型", config.model)
    table.add_row("报告 ID", result.report_id)

    # 成功率 — 只统计 baseline 证据
    baseline_ev = [e for e in result.evidence if e.evidence_type == "baseline"]
    baseline_success = [e for e in baseline_ev if e.success]
    if baseline_ev:
        table.add_row("正常请求成功率", f"{len(baseline_success)}/{len(baseline_ev)}")
    else:
        table.add_row("正常请求成功率", "N/A")

    # 返回模型集合
    models = {e.response_model for e in result.evidence if e.response_model}
    table.add_row("返回模型集合", ", ".join(sorted(models)) if models else "无")

    # 无效模型检查
    invalid_findings = [f for f in result.findings if f.rule_id == "LLMTRACE-INV-001"]
    if invalid_findings:
        inference_text = invalid_findings[0].inferences[0][:80] if invalid_findings[0].inferences else "N/A"
        table.add_row("无效模型是否被拒绝", inference_text)
    else:
        table.add_row("无效模型是否被拒绝", "N/A")

    # 流式
    stream_findings = [f for f in result.findings if f.rule_id == "LLMTRACE-STR-001"]
    if stream_findings:
        table.add_row("流式接口状态", stream_findings[0].status.value)
    else:
        table.add_row("流式接口状态", "N/A")

    # Token 信息完整度
    token_evidence = [e for e in result.evidence if e.input_tokens is not None and e.output_tokens is not None]
    if result.evidence:
        table.add_row("Token 信息完整度", f"{len(token_evidence)}/{len(result.evidence)}")
    else:
        table.add_row("Token 信息完整度", "N/A")

    # 请求 ID 完整度
    rid_evidence = [e for e in result.evidence if e.response_id is not None]
    if result.evidence:
        table.add_row("请求 ID 完整度", f"{len(rid_evidence)}/{len(result.evidence)}")
    else:
        table.add_row("请求 ID 完整度", "N/A")

    # 风险等级
    risk_color = {
        RiskLevel.LOW: "green",
        RiskLevel.MEDIUM: "yellow",
        RiskLevel.HIGH: "red",
        RiskLevel.INCONCLUSIVE: "dim",
    }
    table.add_row(
        "风险等级",
        f"[{risk_color.get(result.risk_level, 'white')}]{result.risk_level.value}[/]",
    )

    _console.print(table)

    if result.findings:
        _console.print()
        findings_table = Table(title="探针结果")
        findings_table.add_column("探针", style="cyan")
        findings_table.add_column("状态", style="white")
        findings_table.add_column("严重程度", style="white")
        for f in result.findings:
            status_color = {
                "pass": "green",
                "fail": "red",
                "warn": "yellow",
                "error": "red",
                "skipped": "dim",
            }
            findings_table.add_row(
                f.probe_name,
                f"[{status_color.get(f.status.value, 'white')}]{f.status.value}[/]",
                f.severity.value,
            )
        _console.print(findings_table)


def print_error(message: str, step: str, partial: bool = False) -> None:
    """打印错误信息."""
    _console.print()
    _console.print(Panel.fit(f"[red]错误: {message}[/]", title=f"步骤: {step}"))
    if partial:
        _console.print("[yellow]已生成部分报告，请检查输出目录。[/]")


def describe_error(exc: BaseException, api_key_env: str | None = None) -> str:
    """把底层异常映射为对用户可行动的中文提示（产品化 §Error）。

    已知类别（连接 / 超时 / HTTP 状态 / 沙箱）给出可执行建议；未知异常
    回退到原始信息，由上层 SecretScrubber 统一脱敏后展示。
    """
    from llmtrace.adapters.code_execution import CodeExecutionError, SandboxUnavailableError

    if isinstance(exc, SandboxUnavailableError):
        return (
            "代码沙箱不可用（Docker 未就绪）。请确认 Docker 已启动；"
            "使用 --no-streaming/--dry-run 只做协议层检查，或先运行 llmtrace doctor 诊断。"
        )
    if isinstance(exc, CodeExecutionError):
        return f"代码执行失败：{exc}"

    try:
        import httpx
    except ImportError:  # pragma: no cover
        return str(exc)

    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code in (401, 403):
            hint = f"鉴权失败（HTTP {code}）。请确认 {api_key_env or 'API Key'} 有效、鉴权头格式正确且未被目标拒绝。"
        elif code == 404:
            hint = "端点或模型不存在（HTTP 404）。请检查 --base-url 路径与 --model 名称是否正确。"
        elif code == 429:
            hint = "触发速率限制（HTTP 429）。请稍后重试；或用 --repeat/-r 调低探测频率、提高 --timeout。"
        else:
            hint = f"目标返回 HTTP {code}。请结合报告中的协议风险结论排查。"
        return hint

    timeout_types: tuple[type[BaseException], ...] = (httpx.TimeoutException,)
    try:
        from httpx import ConnectError, ConnectTimeout, ReadTimeout, WriteTimeout

        timeout_types += (ConnectTimeout, ReadTimeout, WriteTimeout)
        connect_types: tuple[type[BaseException], ...] = (ConnectError,)
    except ImportError:  # pragma: no cover
        connect_types = (httpx.ConnectError,)

    if isinstance(exc, connect_types):
        return (
            "无法连接到目标端点。请检查 --base-url 是否正确、网络/代理是否可达；使用 llmtrace doctor 排除本地环境问题。"
        )
    if isinstance(exc, timeout_types) or type(exc).__name__ == "TimeoutError":
        return "请求超时。端点响应过慢，可用 --timeout 提高超时上限，或稍后重试。"

    if isinstance(exc, httpx.InvalidURL):
        return "base-url 格式不合法。示例: https://api.example.com/v1"

    return str(exc)


def print_dry_run(config_summary: dict[str, str]) -> None:
    """打印 dry-run 执行计划."""
    _console.print()
    _console.print(Panel.fit("Dry Run - 执行计划", style="bold blue"))
    table = Table(title="计划摘要")
    table.add_column("项目", style="cyan")
    table.add_column("值", style="white")
    for key, value in config_summary.items():
        table.add_row(key, str(value))
    _console.print(table)
    _console.print("[yellow]未发送任何请求。[/]")


def print_compare_result(result: dict[str, object]) -> None:
    """打印比较结果."""
    _console.print()
    _console.print(Panel.fit("LLMTrace 报告比较", style="bold blue"))

    table = Table(title="比较摘要")
    table.add_column("项目", style="cyan")
    table.add_column("值", style="white")

    for key, value in result.items():
        if key == "warnings":
            continue
        if isinstance(value, list):
            table.add_row(key, str(value[:3]))
        else:
            table.add_row(key, str(value))

    _console.print(table)

    warnings = result.get("warnings")
    if isinstance(warnings, list) and warnings:
        _console.print()
        _console.print("[yellow]警告:[/]")
        for w in warnings:
            _console.print(f"  [yellow]- {w}[/]")


def print_unified_summary(result: object, artifacts: dict[str, str]) -> None:
    """Print the unified ``llmtrace run`` summary.

    ``artifacts`` maps logical artifact names to their on-disk paths.
    """
    _console.print()
    _console.print(Panel.fit("LLMTrace Unified Audit", style="bold blue"))

    plan = result.plan  # type: ignore[attr-defined]
    table = Table(title="执行摘要")
    table.add_column("项目", style="cyan")
    table.add_column("值", style="white")

    table.add_row("Target", str(plan.target_id))
    protocol = "N/A"
    if result.protocol_audit is not None:  # type: ignore[attr-defined]
        protocol = str(result.protocol_audit.config.protocol.value)  # type: ignore[attr-defined]
    table.add_row("协议", protocol)
    table.add_row("声明模型", plan.candidate_model_id)
    table.add_row("Execution ID", str(result.execution_id))  # type: ignore[attr-defined]
    table.add_row("状态", str(result.status.value))  # type: ignore[attr-defined]

    if result.protocol_audit is not None:  # type: ignore[attr-defined]
        table.add_row("协议风险", str(result.protocol_audit.risk_level.value))  # type: ignore[attr-defined]

    if result.capability_profile is not None:  # type: ignore[attr-defined]
        profile = result.capability_profile  # type: ignore[attr-defined]
        is_calibrated = profile.calibration is not None
        if is_calibrated and profile.calibrated_total_score is not None:
            table.add_row("Calibrated Score", f"{profile.calibrated_total_score:.1f} / 100")
        table.add_row("Coverage", f"{profile.coverage_weight:.2f}")
        for d in profile.dimensions:
            if is_calibrated and d.calibrated_score is not None:
                score_text = f"{d.calibrated_score:.1f} (cal) / {d.raw_normalized_score:.4f} (raw)"
                table.add_row(f"  {d.dimension.value}", score_text)
            else:
                table.add_row(f"  {d.dimension.value}", f"{d.raw_normalized_score:.4f} (raw)")

    drift_text = "no baseline"
    if result.behavior_drift is not None:  # type: ignore[attr-defined]
        drift_text = result.behavior_drift.drift_level.value  # type: ignore[attr-defined]
    table.add_row("Behavior Drift", drift_text)

    measurement = getattr(result, "measurement_summary", None)
    if measurement is not None:
        table.add_row(
            "Benchmark 测量",
            f"{measurement.graded_item_count}/{measurement.total_item_count} graded, "
            f"{measurement.failure_item_count} failure, {measurement.ungradable_item_count} ungradable",
        )
        table.add_row(
            "测量覆盖率",
            f"grading {measurement.grading_coverage:.0%} / execution {measurement.execution_coverage:.0%}",
        )
    elif result.protocol_audit is not None and getattr(result, "benchmark_runs", None):  # type: ignore[attr-defined]
        table.add_row("Benchmark 测量", "unavailable")

    ref_text = "compared" if result.reference_comparison is not None else "unavailable"  # type: ignore[attr-defined]
    table.add_row("Reference", ref_text)
    table.add_row("请求数", f"planned {plan.planned_requests}")

    # ---- Confidence / Routing / Perf（§34–§42 v1 确定性规则；实验性标签） ----
    confidence = ConfidencePolicy.create_v1().assess(
        measurement=getattr(result, "measurement_summary", None),
        capability_profile=getattr(result, "capability_profile", None),
    )
    table.add_row("Confidence", confidence.level.value)

    routing: object | None = None
    snapshot = getattr(result, "behavior_snapshot", None)
    if snapshot is not None:
        routing = assess_routing_stability(snapshot)
        table.add_row("Routing Stability", str(routing.level.value))

    # Perf：墙钟耗时 + 行为样本延迟 / 输出 token 均值（若可计算）。
    finished_at = getattr(result, "finished_at", None)
    if finished_at is not None:
        elapsed = finished_at - result.started_at  # type: ignore[attr-defined]
        table.add_row("耗时", f"{elapsed.total_seconds():.1f}s")
    if snapshot is not None:
        snapshot_items = list(snapshot.items)
        latencies = [it.latency_ms for it in snapshot_items if it.latency_ms is not None]
        outputs = [it.output_tokens for it in snapshot_items if it.output_tokens is not None]
        if latencies:
            table.add_row("平均延迟", f"{sum(latencies) / len(latencies):.0f} ms / 请求")
        if outputs:
            table.add_row("平均输出 Token", f"{sum(outputs) / len(outputs):.0f} / 请求")

    _console.print(table)
    _console.print()

    is_calibrated = (
        result.capability_profile is not None  # type: ignore[attr-defined]
        and getattr(result.capability_profile, "calibration", None) is not None  # type: ignore[attr-defined]
    )
    if is_calibrated:
        _console.print(
            "[bold green]CALIBRATED：[/][green]capability 分数已经过 Reference Calibration，为 0–100 正式评分。[/]"
        )
    else:
        _console.print(
            "[bold yellow]UNCALIBRATED：[/][yellow]capability 分数为 raw / provisional，不是 0–100 正式评分。[/]"
        )

    # ---- 验证注解：Confidence / Routing 的确定性理由（实验性，非统计证明） ----
    verdict_lines: list[str] = []
    if confidence.reasons:
        verdict_lines.append(f"[cyan]Confidence ({confidence.level.value}):[/] {confidence.reasons[0]}")
    if routing is not None:
        routing_attr = getattr(routing, "reasons", ())
        if routing_attr:
            verdict_lines.append(f"[cyan]Routing ({routing.level.value}):[/] {routing_attr[0]}")  # type: ignore[attr-defined]
    if verdict_lines:
        _console.print()
        _console.print("[bold]验证注解（实验性，非统计证明）:[/]")
        for line in verdict_lines:
            _console.print(f"  {line}")

    # ---- Claimed Model Gap（§13/§17：能力差距，不是模型身份识别） ---------
    claimed_gap = getattr(result, "claimed_model_gap", None)
    if claimed_gap is not None:
        _console.print()
        gap_table = Table(title="Claimed Model Comparison")
        gap_table.add_column("项目", style="cyan")
        gap_table.add_column("值", style="white")
        gap_table.add_row("声明模型", claimed_gap.claimed_model_id)
        gap_table.add_row(
            "Trusted Reference",
            f"{claimed_gap.reference_total_score:.1f} / 100"
            f"（{claimed_gap.reference_provider_id} / {claimed_gap.reference_model_id}）",
        )
        gap_table.add_row("Measured Capability", f"{claimed_gap.candidate_total_score:.1f} / 100")
        delta_style = "red" if claimed_gap.total_delta < 0 else "green"
        gap_table.add_row(
            "Capability Gap",
            f"[{delta_style}]{claimed_gap.total_delta:+.1f}[/{delta_style}]",
        )
        for dg in claimed_gap.dimension_gaps:
            dim_style = "red" if dg.delta < 0 else "green"
            gap_table.add_row(
                f"  {dg.dimension.value}",
                f"{dg.candidate_score:.1f} vs {dg.reference_score:.1f}（[{dim_style}]{dg.delta:+.1f}[/{dim_style}]）",
            )
        _console.print(gap_table)
        _console.print()
        _console.print(
            "[bold]解读：[/]被测端点的能力分与声明模型兼容的可信参考配置存在上述差距。这是能力比较，不是模型身份证明。"
        )

    # ---- Model Verification（Task 42 / Task 43）：身份证据，与能力评测并列 ---------
    match = getattr(result, "fingerprint_match", None)
    if match is not None:
        # 局部 import：``analysis.confidence_v2`` 依赖 ``execution.models``，而
        # ``execution.models`` 又经 ``reporting`` 包回到本模块，模块级导入会成环。
        from llmtrace.analysis.confidence_v2 import fingerprint_confidence

        verification = getattr(result, "fingerprint_verification", None)
        snapshot = getattr(result, "fingerprint_snapshot", None)

        _console.print()
        _console.print(Panel.fit("Model Verification · Experimental", style="bold magenta"))

        identity_table = Table(title="行为身份证据（实验性，非密码学证明）")
        identity_table.add_column("项目", style="cyan")
        identity_table.add_column("值", style="white")
        identity_table.add_row("Claimed Model", str(match.claimed_model_id or plan.candidate_model_id))

        set_text = "N/A"
        if plan.fingerprint_set_id:
            set_text = plan.fingerprint_set_id
            if plan.fingerprint_set_version:
                set_text = f"{set_text} v{plan.fingerprint_set_version}"
        identity_table.add_row("Fingerprint Set", set_text)

        # Task 43：只有 reference set、没有 validated decision policy 时必须写
        # UNVALIDATED —— 不借用任何"看起来已验证"的说法。
        decision_policy = getattr(verification, "policy", None)
        if decision_policy is None:
            policy_text = "UNVALIDATED"
        else:
            policy_state = "validated" if decision_policy.validated else "UNVALIDATED"
            policy_text = f"{decision_policy.policy_id} v{decision_policy.policy_version} ({policy_state})"
        identity_table.add_row("Policy", policy_text)

        identity_table.add_row("Claim Consistency", _claim_consistency_label(match.status))

        fingerprint_confidence_component = fingerprint_confidence(
            verification=verification,
            match=match,
            candidate=snapshot.distributions if snapshot is not None else (),
        )
        identity_table.add_row("Fingerprint Confidence", fingerprint_confidence_component.level.value.upper())
        _console.print(identity_table)

        # Task 43：没有 claim verdict 时仍允许 Top-K 排序展示。
        if match.entries:
            _console.print()
            matches_table = Table(title="Top Behavioral Matches")
            matches_table.add_column("#", style="cyan", justify="right")
            matches_table.add_column("Reference", style="white")
            matches_table.add_column("Distance", style="white")
            for rank, entry in enumerate(match.entries[:_TOP_MATCH_DISPLAY_LIMIT], start=1):
                matches_table.add_row(str(rank), entry.model_id, f"distance {entry.distance:.3f}")
            _console.print(matches_table)

        _console.print()
        _console.print("[bold]Important[/]")
        _console.print("Behavioral evidence only.")
        _console.print("This is not cryptographic proof of upstream identity.")

    if artifacts:
        _console.print()
        _console.print("[bold]Artifacts:[/]")
        for name, path in artifacts.items():
            _console.print(f"  [cyan]{name}[/] → {path}")


class CliProgress:
    """``run`` 命令执行进度渲染（施工手册 §Live）。

    - TTY: Rich ``Live`` 进度条，展示阶段 / benchmark 项进度 / 请求数。
    - 非 TTY: 阶段切换打印一行日志，进度事件逐项打印，适合重定向到日志文件。
    - 取消: ``announce_cancel()`` 切换为"正在取消"状态；终态打印已完成项数。

    实现 :class:`llmtrace.execution.progress.ProgressSink`；不写任何文件、
    不发送任何请求，只做终端渲染。
    """

    def __init__(self, *, tty: bool | None = None) -> None:
        self._console = Console()
        # 显式 tty 参数用于测试注入；默认跟随 Rich 的终端检测。
        self._tty = self._console.is_terminal if tty is None else tty
        self._live: Live | None = None
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        # 运行态（由 runner 的 sink 回调 / 信号处理器更新）。
        self.stage: str = STAGE_PREFLIGHT
        self.message: str | None = None
        self.completed: int | None = None
        self.total: int | None = None
        self.requests: int = 0
        self.cancelled: bool = False

    # -- 生命周期 ----------------------------------------------------------

    def start(self) -> None:
        """开始渲染；TTY 进入 Live，非 TTY 打印首行日志。"""
        if not self._tty:
            _console.print("[dim]llmtrace run: 开始执行…[/]")
            return
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}", justify="left"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
        )
        self._task_id = self._progress.add_task("", total=None)
        self._live = Live(self._render(), console=self._console, refresh_per_second=10)
        self._live.start()

    def stop(self) -> None:
        """结束渲染；非 TTY 打印终态行。"""
        if self._live is not None:
            self._live.stop()
            self._live = None
            final_line = self._cancel_line()
            if final_line:
                _console.print(final_line)
            elif not self.cancelled:
                _console.print()
        elif not self._tty:
            final_line = self._cancel_line()
            if final_line:
                _console.print(final_line)

    # -- ProgressSink 回调（runner async 线程，只做轻量状态更新） -----------

    def on_event(self, event: object) -> None:
        """接收 :class:`ProgressEvent`，更新渲染（兼容 ProgressSink 签名）。"""
        payload = getattr(event, "to_dict", lambda: event)()
        event_type = payload.get("type") if isinstance(payload, dict) else None
        if isinstance(payload, dict):
            self.stage = payload.get("stage", self.stage)
            self.message = payload.get("message", self.message)
            if payload.get("completed") is not None:
                self.completed = payload["completed"]
            if payload.get("total") is not None:
                self.total = payload["total"]
            self.requests = payload.get("requests", self.requests)
        if self._live is not None and self._progress is not None and self._task_id is not None:
            # Live 模式：只维护底层状态，由 _refresh 重建 renderable。
            self._refresh()
            return
        if not self._tty:
            self._print_non_tty_line(event_type)

    def announce_cancel(self) -> None:
        """第一次 Ctrl+C：标记取消（不打印、不阻塞信号处理器）。"""
        self.cancelled = True
        self.message = "正在取消… 等待当前项完成后退出（再按一次 Ctrl+C 强制退出）"
        self._refresh()

    # -- 内部 ---------------------------------------------------------------

    def _refresh(self) -> None:
        live = self._live
        progress = self._progress
        task_id = self._task_id
        if live is not None and progress is not None and task_id is not None:
            progress.update(
                task_id,
                description=self._description(),
                completed=self.completed or 0,
                total=self.total if self.total is not None else None,
            )
            live.update(self._render())

    def _render(self) -> RenderableType:
        if self._progress is None:
            return Text("")
        status = "请求: " + str(self.requests)
        status_render = Text(status, style="yellow") if self.cancelled else status
        return Panel(Group(self._progress, status_render), title="llmtrace run", style="green")

    def _description(self) -> str:
        label = _STAGE_LABELS.get(self.stage, self.stage)
        suffix = ""
        if self.message:
            suffix = f" - {self.message}"
        return f"{label}{suffix}"

    def _cancel_line(self) -> str | None:
        """取消后的汇总行：已完成项数 + 请求数。"""
        if not self.cancelled:
            return None
        progress_text = f"{self.completed}/{self.total}" if self.total else f"{self.completed or 0}/?"
        return f"[bold yellow]已取消:[/] 已完成 {progress_text} 个 benchmark 项，已发出 {self.requests} 个请求"

    def _print_non_tty_line(self, event_type: str | None) -> None:
        """非 TTY：阶段切换与逐项进度打印为可读日志行。"""
        line = f"[llmtrace] {self._description()}"
        if event_type in {"progress", "done", "failed", "cancelled"}:
            if self.completed is not None and self.total:
                line = f"[llmtrace] benchmark 进度 {self.completed}/{self.total}（请求 {self.requests}）"
            if event_type == "done":
                line = "[llmtrace] 执行完成"
            elif event_type in {"failed", "cancelled"}:
                line = f"[llmtrace] 执行{'失败' if event_type == 'failed' else '已取消'}"
        _console.print(line)


__all__ = [
    "CliProgress",
    "print_audit_summary",
    "print_compare_result",
    "print_dry_run",
    "print_error",
    "print_unified_summary",
]
