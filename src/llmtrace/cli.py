"""LLMTrace CLI 入口."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast

import typer

from llmtrace.adapters.code_execution import SandboxUnavailableError
from llmtrace.analysis.drift import compare_reports
from llmtrace.config import AuditConfig, AuthStyle, Protocol
from llmtrace.constants import (
    DEFAULT_REPEAT_COUNT,
    DEFAULT_TIMEOUT,
    MAX_OUTPUT_TOKENS_DEFAULT,
    MAX_RESPONSE_BYTES_DEFAULT,
)
from llmtrace.execution.artifacts import RunArtifactRepository
from llmtrace.execution.planner import build_unified_execution_plan, derive_target_id, sanitize_target_id
from llmtrace.execution.protocol_audit import ProtocolAuditExecutor, build_audit_plan
from llmtrace.execution.runner import UnifiedAuditRunner
from llmtrace.models.evidence import HTTPEvidence
from llmtrace.models.findings import FindingResult
from llmtrace.providers.base import BaseProvider
from llmtrace.reporting.console import (
    print_audit_summary,
    print_compare_result,
    print_dry_run,
    print_error,
    print_unified_summary,
)
from llmtrace.reporting.html_report import generate_html_report
from llmtrace.reporting.json_report import generate_json_report
from llmtrace.security.redaction import check_api_key

app = typer.Typer(
    name="llmtrace",
    help="LLMTrace - 模型寻迹：面向第三方 AI API 的黑盒模型审计工具",
    add_completion=False,
)


def _create_provider(config: AuditConfig, api_key: str) -> BaseProvider:
    """根据协议创建 Provider（兼容旧调用，单一实现见 providers.factory）."""
    from llmtrace.providers.factory import create_provider

    return create_provider(config, api_key)


def _build_audit_plan(config: AuditConfig) -> list[dict[str, object]]:
    """构建审计计划，用于 dry-run 和实际执行."""

    return build_audit_plan(config)


def _validate_evidence_refs(findings: list[FindingResult], evidence_list: list[HTTPEvidence]) -> None:
    """验证所有 evidence_refs 都能在 evidence 集合中找到."""
    evidence_ids = {str(e.evidence_id) for e in evidence_list}
    for f in findings:
        for ref in f.evidence_refs:
            if ref not in evidence_ids:
                raise ValueError(f"证据引用 '{ref}' (探针: {f.probe_name}) 在证据集合中找不到。")


def _check_duplicate_evidence_ids(evidence_list: list[HTTPEvidence]) -> None:
    """检查是否有重复的 evidence_id."""
    seen: set[str] = set()
    for ev in evidence_list:
        eid = str(ev.evidence_id)
        if eid in seen:
            raise ValueError(f"重复的 evidence_id: {eid}")
        seen.add(eid)


@app.command()
def audit(
    protocol: str = typer.Option(..., "--protocol", "-p", help="协议类型: openai 或 anthropic"),
    base_url: str = typer.Option(..., "--base-url", "-u", help="API Base URL"),
    model: str = typer.Option(..., "--model", "-m", help="模型名称"),
    api_key_env: str = typer.Option(..., "--api-key-env", "-k", help="API Key 环境变量名"),
    auth_style: str = typer.Option("auto", "--auth-style", help="鉴权方式: auto, bearer, x-api-key, both"),
    repeat: int = typer.Option(DEFAULT_REPEAT_COUNT, "--repeat", "-r", help="重复次数 (1-10)"),
    timeout: float = typer.Option(DEFAULT_TIMEOUT, "--timeout", "-t", help="请求超时(秒)"),
    max_output_tokens: int = typer.Option(MAX_OUTPUT_TOKENS_DEFAULT, "--max-tokens", help="最大输出 Token"),
    max_response_bytes: int = typer.Option(
        MAX_RESPONSE_BYTES_DEFAULT, "--max-response-bytes", help="最大响应体保存字节数"
    ),
    check_streaming: bool = typer.Option(True, "--streaming/--no-streaming", help="是否检查流式接口"),
    output_dir: Path = typer.Option(Path("reports"), "--output-dir", "-o", help="输出目录"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只显示执行计划，不发送请求"),
    non_interactive: bool = typer.Option(False, "--yes", "-y", help="非交互模式，自动确认"),
    debug: bool = typer.Option(False, "--debug", help="显示完整异常堆栈"),
) -> None:
    """执行 API 审计."""
    config = AuditConfig(
        protocol=Protocol(protocol),
        base_url=base_url,
        model=model,
        api_key_env=api_key_env,
        auth_style=AuthStyle(auth_style),
        repeat_count=repeat,
        timeout=timeout,
        max_output_tokens=max_output_tokens,
        max_response_bytes=max_response_bytes,
        check_streaming=check_streaming,
        output_dir=output_dir,
    )

    # 构建审计计划
    plan = _build_audit_plan(config)
    total_requests = sum(int(cast(int, item["count"])) for item in plan)

    if dry_run:
        print_dry_run(
            {
                "协议": config.protocol.value,
                "Base URL": config.base_url,
                "模型": config.model,
                "重复次数": str(config.repeat_count),
                "预计调用次数": str(total_requests),
                "最大输出 Token": str(config.max_output_tokens),
                "流式检查": "是" if config.check_streaming else "否",
                "无效模型调用": "是",
            }
        )
        return

    api_key = check_api_key(config.api_key_env)
    if api_key is None:
        print_error(f"环境变量 {config.api_key_env} 不存在或为空", "配置检查", partial=False)
        raise typer.Exit(code=1)

    provider = _create_provider(config, api_key)
    try:
        outcome = asyncio.run(ProtocolAuditExecutor(config, provider).run())
    except Exception as e:
        if debug:
            import traceback

            traceback.print_exc()
        print_error(str(e), "审计执行", partial=True)
        raise typer.Exit(code=1)

    result = outcome.result

    json_path = config.output_dir / f"{result.report_id}.json"
    html_path = config.output_dir / f"{result.report_id}.html"

    try:
        generate_json_report(result, json_path)
    except Exception:
        if debug:
            import traceback

            traceback.print_exc()

    try:
        generate_html_report(result, html_path)
    except Exception:
        if debug:
            import traceback

            traceback.print_exc()

    print_audit_summary(result)

    from rich.console import Console

    console = Console()
    console.print()
    console.print(f"JSON 报告: [cyan]{json_path}[/]")
    console.print(f"HTML 报告: [cyan]{html_path}[/]")


@app.command()
def run(
    protocol: str = typer.Option(..., "--protocol", "-p", help="协议类型: openai 或 anthropic"),
    base_url: str = typer.Option(..., "--base-url", "-u", help="API Base URL"),
    model: str = typer.Option(..., "--model", "-m", help="声明模型名称"),
    api_key_env: str = typer.Option(..., "--api-key-env", "-k", help="API Key 环境变量名"),
    auth_style: str = typer.Option("auto", "--auth-style", help="鉴权方式: auto, bearer, x-api-key, both"),
    target_id: str = typer.Option(None, "--target-id", help="稳定 target 标识（缺省自动派生）"),
    repeat: int = typer.Option(DEFAULT_REPEAT_COUNT, "--repeat", "-r", help="协议探针重复次数 (1-10)"),
    timeout: float = typer.Option(DEFAULT_TIMEOUT, "--timeout", "-t", help="请求超时(秒)"),
    check_streaming: bool = typer.Option(True, "--streaming/--no-streaming", help="是否检查流式接口"),
    output_dir: Path = typer.Option(Path("reports"), "--output-dir", "-o", help="artifact 根目录"),
    reference_snapshot: Path = typer.Option(None, "--reference-snapshot", help="ReferenceSnapshot JSON 路径"),
    baseline_snapshot: Path = typer.Option(None, "--baseline-snapshot", help="显式基线 BehaviorRunSnapshot JSON 路径"),
    compare_latest: bool = typer.Option(
        True, "--compare-latest/--no-compare-latest", help="自动与最新兼容历史运行比较"
    ),
    max_wall_seconds: float = typer.Option(None, "--max-wall-seconds", help="整体墙钟超时(秒)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只显示执行计划，不发送请求"),
    non_interactive: bool = typer.Option(False, "--yes", "-y", help="跳过确认"),
    debug: bool = typer.Option(False, "--debug", help="显示完整异常堆栈"),
) -> None:
    """执行统一审计：协议审计 + Quick Suite 32 题 + 能力/行为/工件."""
    import sys

    config = AuditConfig(
        protocol=Protocol(protocol),
        base_url=base_url,
        model=model,
        api_key_env=api_key_env,
        auth_style=AuthStyle(auth_style),
        repeat_count=repeat,
        timeout=timeout,
        check_streaming=check_streaming,
        output_dir=output_dir,
    )

    resolved_target = (
        sanitize_target_id(target_id) if target_id else derive_target_id(config.protocol.value, config.base_url)
    )
    plan = build_unified_execution_plan(config, target_id=resolved_target)

    if dry_run:
        print_dry_run(
            {
                "Target": resolved_target,
                "协议": config.protocol.value,
                "声明模型": config.model,
                "Suite": f"{plan.suite_id} {plan.suite_version}",
                "协议探针请求": str(plan.protocol_probe_requests),
                "Benchmark 请求": str(plan.benchmark_requests),
                "总请求上限": str(plan.maximum_requests),
                "输出 Token 上限": str(plan.maximum_output_token_ceiling),
                "预计费用": "unknown",
                "需要安全 Sandbox": "是",
                "参考对比": "是" if reference_snapshot else "否",
                "历史对比": "是" if compare_latest else "否",
            }
        )
        return

    api_key = check_api_key(config.api_key_env)
    if api_key is None:
        print_error(f"环境变量 {config.api_key_env} 不存在或为空", "配置检查", partial=False)
        raise typer.Exit(code=1)

    # Confirmation
    if not non_interactive:
        if not sys.stdin.isatty():
            print_error("非交互式执行需要 --yes", "确认", partial=False)
            raise typer.Exit(code=1)
        print_dry_run(
            {
                "Target": resolved_target,
                "总请求上限": str(plan.maximum_requests),
                "输出 Token 上限": str(plan.maximum_output_token_ceiling),
                "预计费用": "unknown",
            }
        )
        import typer as _typer

        if not _typer.confirm(f"本运行可能发送最多 {plan.maximum_requests} 个请求。是否继续？"):
            raise typer.Exit(code=0)

    repository = RunArtifactRepository(output_dir)
    try:
        runner = UnifiedAuditRunner(
            config,
            api_key=api_key,
            target_id=resolved_target,
            repository=repository,
            compare_latest=compare_latest,
            baseline_snapshot_path=baseline_snapshot,
            reference_snapshot_path=reference_snapshot,
            max_wall_seconds=max_wall_seconds,
        )
    except SandboxUnavailableError as exc:
        print_error(str(exc), "预检", partial=False)
        raise typer.Exit(code=1)

    try:
        result = asyncio.run(runner.run())
    except KeyboardInterrupt:
        raise typer.Exit(code=130)
    except Exception as exc:
        if debug:
            import traceback

            traceback.print_exc()
        print_error(str(exc), "统一执行", partial=True)
        raise typer.Exit(code=1)

    artifact_paths = {
        "manifest.json": str(output_dir / "runs" / result.execution_id / "manifest.json"),
        "report.json": str(output_dir / "runs" / result.execution_id / "report.json"),
        "report.html": str(output_dir / "runs" / result.execution_id / "report.html"),
    }
    print_unified_summary(result, artifact_paths)


@app.command()
def compare(
    report_a: Path = typer.Argument(..., help="第一份报告路径"),
    report_b: Path = typer.Argument(..., help="第二份报告路径"),
    additional_reports: list[Path] = typer.Argument(None, help="额外报告路径"),
) -> None:
    """比较多份审计报告."""
    paths = [report_a, report_b]
    if additional_reports:
        paths.extend(additional_reports)

    reports_data = []
    for p in paths:
        if not p.exists():
            typer.echo(f"错误: 报告文件不存在: {p}", err=True)
            raise typer.Exit(code=1)
        try:
            with open(p, encoding="utf-8") as f:
                reports_data.append(json.load(f))
        except json.JSONDecodeError as e:
            typer.echo(f"错误: 无法解析报告 {p}: {e}", err=True)
            raise typer.Exit(code=1)

    result = compare_reports(reports_data)

    print_compare_result(
        {
            "报告数": result.report_count,
            "Endpoint": result.endpoints[0] if result.endpoints else "N/A",
            "声称模型": result.claimed_models[0] if result.claimed_models else "N/A",
            "成功率": [f"{r:.0%}" for r in result.success_rates],
            "延迟中位数(ms)": [f"{m:.0f}" for m in result.latency_medians_ms],
            "返回模型集合": [str(s) for s in result.response_model_sets],
            "指纹集合": [str(s) for s in result.fingerprint_sets],
            "Token 字段存在率": [f"{r:.0%}" for r in result.token_field_rates],
            "请求 ID 存在率": [f"{r:.0%}" for r in result.request_id_rates],
            "风险等级": result.risk_levels,
            "漂移判断": result.drift_level.value,
            "漂移说明": result.drift_notes,
            "版本不匹配": result.version_mismatch,
            "警告": result.warnings,
        }
    )


@app.command()
def inspect(
    report_path: Path = typer.Argument(..., help="JSON 报告路径"),
) -> None:
    """查看 JSON 报告摘要."""
    if not report_path.exists():
        typer.echo(f"错误: 报告文件不存在: {report_path}", err=True)
        raise typer.Exit(code=1)

    try:
        with open(report_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        typer.echo(f"错误: 无法解析报告: {e}", err=True)
        raise typer.Exit(code=1)

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    console.print()
    console.print(Panel.fit("LLMTrace 报告查看", style="bold blue"))

    meta = data.get("meta", {})
    config = data.get("config", {})

    table = Table(title="报告元数据")
    table.add_column("项目", style="cyan")
    table.add_column("值", style="white")

    table.add_row("报告 ID", str(data.get("report_id", "N/A")))
    table.add_row("LLMTrace 版本", str(meta.get("llmtrace_version", "N/A")))
    table.add_row("测试套件版本", str(meta.get("test_suite_version", "N/A")))
    table.add_row("生成时间", str(meta.get("utc_time", "N/A")))
    table.add_row("Python", str(meta.get("python_version", "N/A")))
    table.add_row("平台", str(meta.get("platform", "N/A")))
    table.add_row("Endpoint", str(config.get("base_url", "N/A")))
    table.add_row("协议", str(config.get("protocol", "N/A")))
    table.add_row("模型", str(config.get("model", "N/A")))
    table.add_row("风险等级", str(data.get("risk_level", "N/A")))
    h = str(meta.get("content_hash", ""))
    table.add_row("内容哈希", h[:16] + "..." if len(h) > 16 else h)

    console.print(table)

    # 显示探针结果
    findings = data.get("findings", [])
    if findings:
        console.print()
        findings_table = Table(title="探针结果")
        findings_table.add_column("探针", style="cyan")
        findings_table.add_column("状态", style="white")
        findings_table.add_column("严重程度", style="white")
        for f in findings:
            findings_table.add_row(
                str(f.get("probe_name", "N/A")),
                str(f.get("status", "N/A")),
                str(f.get("severity", "N/A")),
            )
        console.print(findings_table)

    # 证据概要
    evidence = data.get("evidence", [])
    if evidence:
        console.print()
        ev_table = Table(title="证据概要")
        ev_table.add_column("#", style="white")
        ev_table.add_column("类型", style="cyan")
        ev_table.add_column("请求模型", style="cyan")
        ev_table.add_column("返回模型", style="white")
        ev_table.add_column("HTTP", style="white")
        ev_table.add_column("成功", style="white")
        for i, ev in enumerate(evidence, 1):
            ev_table.add_row(
                str(i),
                str(ev.get("evidence_type", "N/A")),
                str(ev.get("request_model", "N/A")),
                str(ev.get("response_model", "N/A")),
                str(ev.get("http_status", "N/A")),
                "是" if ev.get("success") else "否",
            )
        console.print(ev_table)

    # 执行元数据（v0.3-E）
    execution = data.get("execution")
    if execution:
        console.print()
        exec_table = Table(title="执行元数据")
        exec_table.add_column("项目", style="cyan")
        exec_table.add_column("值", style="white")
        for key, value in execution.items():
            exec_table.add_row(str(key), str(value))
        console.print(exec_table)

    # Capability Profile（v0.3-E，明确 uncalibrated）
    capability = data.get("capability_profile")
    if capability:
        console.print()
        cap_table = Table(title="Capability Profile (raw / uncalibrated)")
        cap_table.add_column("维度", style="cyan")
        cap_table.add_column("状态", style="white")
        cap_table.add_column("Raw Score", style="white")
        cap_table.add_column("Coverage", style="white")
        for d in capability.get("dimensions", []):
            cap_table.add_row(
                str(d.get("dimension", "N/A")),
                str(d.get("status", "N/A")),
                f"{d.get('raw_normalized_score', 0.0):.4f}",
                f"{d.get('task_coverage', 0.0):.2f}",
            )
        console.print(cap_table)
        console.print("[yellow]UNCALIBRATED：以上为 raw / provisional 分数，不是 0–100 正式能力分。[/]")

    # Reference Comparison（v0.3-C）
    reference = data.get("reference_comparison")
    if reference:
        console.print()
        ref_table = Table(title="Reference Comparison")
        ref_table.add_column("项目", style="cyan")
        ref_table.add_column("值", style="white")
        ref_table.add_row("Reference Snapshot", str(reference.get("reference_snapshot", "N/A")))
        ref_table.add_row("Suite", f"{reference.get('suite_id', 'N/A')} {reference.get('suite_version', '')}")
        console.print(ref_table)

    # Behavior Drift（v0.3-D）
    behavior = data.get("behavior_drift")
    if behavior:
        console.print()
        bh_table = Table(title="Behavior Drift")
        bh_table.add_column("项目", style="cyan")
        bh_table.add_column("值", style="white")
        bh_table.add_row("Drift Level", str(behavior.get("drift_level", "N/A")))
        bh_table.add_row("Policy", f"{behavior.get('policy_id', 'N/A')} {behavior.get('policy_version', '')}")
        bh_table.add_row(
            "Graded Overlap",
            f"{behavior.get('graded_overlap_count', '?')} / {behavior.get('total_items', '?')}",
        )
        console.print(bh_table)


if __name__ == "__main__":
    app()
