"""终端控制台输出."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llmtrace.models.audit import AuditResult, RiskLevel
from llmtrace.security.redaction import redact_url

_console = Console()


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

    table.add_row("Target", str(result.target_id))  # type: ignore[attr-defined]
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
                f"{dg.candidate_score:.1f} vs {dg.reference_score:.1f}"
                f"（[{dim_style}]{dg.delta:+.1f}[/{dim_style}]）",
            )
        _console.print(gap_table)
        _console.print()
        _console.print(
            "[bold]解读：[/]被测端点的能力分与声明模型兼容的可信参考配置存在上述差距。"
            "这是能力比较，不是模型身份证明。"
        )

    if artifacts:
        _console.print()
        _console.print("[bold]Artifacts:[/]")
        for name, path in artifacts.items():
            _console.print(f"  [cyan]{name}[/] → {path}")
