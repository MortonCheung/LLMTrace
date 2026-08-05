"""LLMTrace CLI 入口."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import typer

from llmtrace.analysis.drift import compare_reports
from llmtrace.analysis.risk import analyze_risk
from llmtrace.analysis.schema_fingerprint import generate_schema_fingerprint
from llmtrace.config import AuditConfig, AuthStyle, Protocol
from llmtrace.constants import (
    DEFAULT_REPEAT_COUNT,
    DEFAULT_TIMEOUT,
    MAX_OUTPUT_TOKENS_DEFAULT,
    MAX_RESPONSE_BYTES_DEFAULT,
)
from llmtrace.models.audit import AuditResult, RiskLevel
from llmtrace.models.evidence import HTTPEvidence
from llmtrace.models.findings import FindingResult
from llmtrace.probes.baseline import BaselineProbe
from llmtrace.probes.connectivity import ConfigPrecheckProbe, ConnectivityProbe
from llmtrace.probes.invalid_model import InvalidModelProbe
from llmtrace.probes.metadata import MetadataProbe
from llmtrace.probes.model_catalog import ModelCatalogProbe
from llmtrace.probes.stability import StabilityProbe
from llmtrace.probes.streaming import StreamingProbe
from llmtrace.providers.anthropic_compatible import AnthropicCompatibleProvider
from llmtrace.providers.base import BaseProvider
from llmtrace.providers.openai_compatible import OpenAICompatibleProvider
from llmtrace.reporting.console import (
    print_audit_summary,
    print_compare_result,
    print_dry_run,
    print_error,
)
from llmtrace.reporting.html_report import generate_html_report
from llmtrace.reporting.json_report import generate_json_report
from llmtrace.security.redaction import check_api_key
from llmtrace.utilities.hashing import short_id
from llmtrace.utilities.time import format_file_time, utc_now
from llmtrace.utilities.version import get_llmtrace_version, get_platform, get_python_version

app = typer.Typer(
    name="llmtrace",
    help="LLMTrace - 模型寻迹：面向第三方 AI API 的黑盒模型审计工具",
    add_completion=False,
)


def _create_provider(config: AuditConfig, api_key: str) -> BaseProvider:
    """根据协议创建 Provider."""
    if config.protocol == Protocol.OPENAI:
        return OpenAICompatibleProvider(config, api_key)
    elif config.protocol == Protocol.ANTHROPIC:
        return AnthropicCompatibleProvider(config, api_key)
    else:
        raise ValueError(f"不支持的协议: {config.protocol}")


def _build_audit_plan(config: AuditConfig) -> list[dict[str, object]]:
    """构建审计计划，用于 dry-run 和实际执行."""
    plan: list[dict[str, object]] = []

    # 1. 配置预检（0 次请求）
    plan.append({"probe": "配置预检", "request_type": "none", "count": 0, "model_type": "N/A", "streaming": False})

    # 2. 连接与鉴权（1 次请求，声明模型）
    plan.append(
        {
            "probe": "连接与鉴权",
            "request_type": "completion",
            "count": 1,
            "model_type": "声明模型",
            "streaming": False,
        }
    )

    # 3. 模型列表（1 次 GET 请求）
    plan.append({"probe": "模型列表", "request_type": "GET", "count": 1, "model_type": "N/A", "streaming": False})

    # 4. 正常基线（repeat_count 次请求，声明模型）
    plan.append(
        {
            "probe": "正常基线",
            "request_type": "completion",
            "count": config.repeat_count,
            "model_type": "声明模型",
            "streaming": False,
        }
    )

    # 5. 无效模型（1 次请求，随机无效模型）
    plan.append(
        {
            "probe": "无效模型",
            "request_type": "completion",
            "count": 1,
            "model_type": "随机无效模型",
            "streaming": False,
        }
    )

    # 6. 流式一致性（1 次非流式 + 1 次流式）
    if config.check_streaming:
        plan.append(
            {
                "probe": "流式一致性",
                "request_type": "completion+stream",
                "count": 2,
                "model_type": "声明模型",
                "streaming": True,
            }
        )

    # 7-8. 元数据完整性和会话稳定性（0 次额外请求，分析已有证据）
    plan.append(
        {"probe": "元数据完整性", "request_type": "analysis", "count": 0, "model_type": "N/A", "streaming": False}
    )
    plan.append(
        {"probe": "会话稳定性", "request_type": "analysis", "count": 0, "model_type": "N/A", "streaming": False}
    )

    return plan


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
        print_error(
            f"环境变量 {config.api_key_env} 不存在或为空",
            "配置检查",
            partial=False,
        )
        if debug:
            raise typer.Exit(code=1)
        raise typer.Exit(code=1)

    provider = _create_provider(config, api_key)
    evidence_list: list[HTTPEvidence] = []
    findings: list[FindingResult] = []

    result = AuditResult(
        config=config,
        llmtrace_version=get_llmtrace_version(),
        python_version=get_python_version(),
        platform=get_platform(),
        report_id=f"llmtrace_{format_file_time(utc_now())}_{short_id(6)}",
        start_time=utc_now(),
    )

    async def _run() -> None:
        async with provider:
            nonlocal findings, evidence_list

            # 1. 配置预检
            precheck = ConfigPrecheckProbe(config, provider)
            outcome = await precheck.run()
            findings.extend(outcome.findings)

            if outcome.findings and outcome.findings[0].status.value == "fail":
                result.findings = findings
                result.risk_level = RiskLevel.INCONCLUSIVE
                result.end_time = utc_now()
                print_error("配置预检失败", "配置预检")
                return

            # 2. 连接与鉴权
            conn = ConnectivityProbe(config, provider)
            outcome = await conn.run()
            findings.extend(outcome.findings)
            evidence_list.extend(outcome.evidence)

            if outcome.findings and outcome.findings[0].status.value == "fail":
                result.findings = findings
                result.risk_level = RiskLevel.INCONCLUSIVE
                result.end_time = utc_now()
                print_error("连接或鉴权失败", "连接与鉴权")
                return

            # 3. 模型列表
            catalog = ModelCatalogProbe(config, provider)
            outcome = await catalog.run()
            findings.extend(outcome.findings)
            evidence_list.extend(outcome.evidence)

            # 获取模型列表数据
            if outcome.evidence:
                list_ev = outcome.evidence[0]
                result.model_list_available = list_ev.success
                # 尝试从 evidence 中提取模型列表
                if list_ev.success and list_ev.response_body_summary:
                    try:
                        data = json.loads(list_ev.response_body_summary)
                        models = data.get("data", [])
                        if isinstance(models, list):
                            result.model_list = [m.get("id", "") for m in models if isinstance(m, dict)]
                            result.model_in_list = config.model in result.model_list
                    except (json.JSONDecodeError, AttributeError):
                        pass

            # 4. 正常基线
            baseline = BaselineProbe(config, provider)
            outcome = await baseline.run()
            findings.extend(outcome.findings)
            evidence_list.extend(outcome.evidence)

            # 5. 无效模型
            invalid = InvalidModelProbe(config, provider)
            outcome = await invalid.run()
            findings.extend(outcome.findings)
            evidence_list.extend(outcome.evidence)

            # 6. 流式一致性
            if config.check_streaming:
                streaming = StreamingProbe(config, provider)
                outcome = await streaming.run()
                findings.extend(outcome.findings)
                evidence_list.extend(outcome.evidence)

            # 7. 元数据完整性
            metadata = MetadataProbe(config, provider)
            meta_result = metadata.analyze(evidence_list)
            findings.append(meta_result)

            # 8. 会话稳定性
            stability = StabilityProbe(config, provider)
            stab_result = stability.analyze(evidence_list)
            findings.append(stab_result)

            # 生成结构指纹
            for ev in evidence_list:
                if ev.response_body_summary:
                    fp = generate_schema_fingerprint(ev.response_body_summary)
                    if fp:
                        result.schema_fingerprints.append(fp)

            # 完整性校验
            _validate_evidence_refs(findings, evidence_list)
            _check_duplicate_evidence_ids(evidence_list)

    try:
        import asyncio

        asyncio.run(_run())
    except Exception as e:
        if debug:
            import traceback

            traceback.print_exc()
        print_error(str(e), "审计执行", partial=True)
        if debug:
            raise typer.Exit(code=1)
        raise typer.Exit(code=1)

    result.evidence = evidence_list
    result.findings = findings
    result.risk_level = analyze_risk(findings)
    result.end_time = utc_now()

    # 生成报告
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

    # 输出报告路径
    from rich.console import Console

    console = Console()
    console.print()
    console.print(f"JSON 报告: [cyan]{json_path}[/]")
    console.print(f"HTML 报告: [cyan]{html_path}[/]")


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


if __name__ == "__main__":
    app()
