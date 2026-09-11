"""LLMTrace CLI 入口."""

from __future__ import annotations

import asyncio
import json
import signal
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import typer

from llmtrace.adapters.code_execution import SandboxUnavailableError
from llmtrace.analysis.drift import compare_reports
from llmtrace.config import AuditConfig, AuthStyle, Protocol
from llmtrace.constants import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_PROTOCOL,
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
from llmtrace.reference import ReferenceCaptureService, ReferenceSetBuilder, ReferenceSetRepository
from llmtrace.reference.capture import ReferenceCaptureStatus
from llmtrace.reference.reference_set import (
    ReferenceSetError,
)
from llmtrace.reporting.console import (
    CliProgress,
    describe_error,
    print_audit_summary,
    print_compare_result,
    print_dry_run,
    print_error,
    print_unified_summary,
)
from llmtrace.reporting.html_report import generate_html_report
from llmtrace.reporting.json_report import generate_json_report
from llmtrace.scoring.errors import ReferenceError, ReferenceNotFoundError
from llmtrace.scoring.reference import ReferenceRepository
from llmtrace.security.redaction import (
    SecretScrubber,
    check_api_key,
    extract_url_secret_values,
    redact_url,
)

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


def _announce_reference_set(set_path: Path) -> None:
    """§9：开始前显示将使用的 ReferenceSet（只读，不重新验证信任链）."""
    from rich.console import Console

    from llmtrace.reference.reference_set import ReferenceSet

    console = Console()
    try:
        reference_set = ReferenceSet.model_validate_json(set_path.read_text(encoding="utf-8"))
    except Exception:
        return
    console.print()
    console.print("[bold cyan]Reference Calibration[/]")
    console.print(f"[bold]{reference_set.reference_set_id}[/] v{reference_set.reference_set_version}")
    console.print(f"Members: {len(reference_set.members)} trusted identities")
    console.print(f"Set: {set_path}")


@app.command()
def audit(
    protocol: str = typer.Option(DEFAULT_PROTOCOL, "--protocol", "-p", help="协议类型: openai 或 anthropic"),
    base_url: str = typer.Option(..., "--base-url", "-u", help="API Base URL"),
    model: str = typer.Option(..., "--model", "-m", help="模型名称"),
    api_key_env: str = typer.Option(
        DEFAULT_API_KEY_ENV, "--api-key-env", "-k", help="API Key 环境变量名（不直接传 key）"
    ),
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
        # Non-debug error output crosses a display boundary — scrub every known
        # secret (API key + base_url credentials) in case the exception echoes it.
        scrubber = SecretScrubber([api_key, *extract_url_secret_values(config.base_url)])
        print_error(scrubber.scrub_text(describe_error(e, api_key_env=config.api_key_env)), "审计执行", partial=True)
        raise typer.Exit(code=1)

    result = outcome.result

    json_path = config.output_dir / f"{result.report_id}.json"
    html_path = config.output_dir / f"{result.report_id}.html"

    # Same serialization-boundary discipline as the unified runner: scrub
    # belongs to canonical serialization (before content_hash), so legacy
    # reports never persist stale-hash content either.
    scrubber = SecretScrubber([api_key, *extract_url_secret_values(config.base_url)])
    try:
        generate_json_report(result, json_path, secret_scrubber=scrubber)
    except Exception:
        if debug:
            import traceback

            traceback.print_exc()

    try:
        generate_html_report(result, html_path, secret_scrubber=scrubber)
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
    protocol: str = typer.Option(DEFAULT_PROTOCOL, "--protocol", "-p", help="协议类型: openai 或 anthropic"),
    base_url: str = typer.Option(..., "--base-url", "-u", help="API Base URL"),
    model: str = typer.Option(..., "--model", "-m", help="声明模型名称"),
    api_key_env: str = typer.Option(
        DEFAULT_API_KEY_ENV, "--api-key-env", "-k", help="API Key 环境变量名（不直接传 key）"
    ),
    auth_style: str = typer.Option("auto", "--auth-style", help="鉴权方式: auto, bearer, x-api-key, both"),
    target_id: str = typer.Option(None, "--target-id", help="稳定 target 标识（缺省自动派生）"),
    repeat: int = typer.Option(DEFAULT_REPEAT_COUNT, "--repeat", "-r", help="协议探针重复次数 (1-10)"),
    timeout: float = typer.Option(DEFAULT_TIMEOUT, "--timeout", "-t", help="请求超时(秒)"),
    check_streaming: bool = typer.Option(True, "--streaming/--no-streaming", help="是否检查流式接口"),
    output_dir: Path = typer.Option(Path("reports"), "--output-dir", "-o", help="artifact 根目录"),
    reference_snapshot: Path = typer.Option(None, "--reference-snapshot", help="ReferenceSnapshot JSON 路径"),
    reference_set: Path = typer.Option(
        None, "--reference-set", help="ReferenceSet JSON 路径（用于 Reference Calibration）"
    ),
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

    reference_set_id: str | None = None
    reference_set_version: str | None = None
    reference_set_content_sha256: str | None = None
    calibration_policy_id: str | None = None
    calibration_policy_version: str | None = None

    # §8–§10: explicit --reference-set wins, otherwise auto-discovery scans the
    # default reference directory.  Priority: explicit > auto discovery > none.
    resolved_reference_set: Path | None = reference_set
    if resolved_reference_set is None:
        from llmtrace.reference.discovery import discover_compatible_sets

        discovered = discover_compatible_sets(RunArtifactRepository(output_dir))
        if len(discovered) == 1:
            resolved_reference_set = discovered[0]
        elif len(discovered) > 1:
            # Never pick at random (§9).  Fail closed: the user must choose.
            from rich.console import Console as _RichConsole

            _RichConsole().print(
                "[yellow]Multiple compatible ReferenceSets found. Specify one with --reference-set.[/]"
            )
            for candidate in discovered:
                _RichConsole().print(f"  {candidate}")
            raise typer.Exit(code=1)

    if resolved_reference_set is not None:
        # Shared validator — identical to the runner's preflight.  A
        # ReferenceSet that cannot support formal calibration is rejected
        # here, before any dry-run or execution proceeds.  Read-only: it
        # never sends HTTP, never creates a provider, never runs candidate
        # code, and never writes an artifact.
        try:
            from llmtrace.reference.validation import validate_reference_set_for_calibration

            context = validate_reference_set_for_calibration(
                set_path=resolved_reference_set,
                artifact_repository=RunArtifactRepository(output_dir),
            )
            reference_set_id = context.reference_set.reference_set_id
            reference_set_version = context.reference_set.reference_set_version
            reference_set_content_sha256 = context.reference_set.content_sha256
            calibration_policy_id = context.calibration_policy.policy_id
            calibration_policy_version = context.calibration_policy.policy_version
        except Exception as exc:
            print_error(str(exc), "reference set 预检", partial=False)
            raise typer.Exit(code=1)

    plan = build_unified_execution_plan(
        config,
        target_id=resolved_target,
        reference_set_id=reference_set_id,
        reference_set_version=reference_set_version,
        reference_set_content_sha256=reference_set_content_sha256,
        calibration_policy_id=calibration_policy_id,
        calibration_policy_version=calibration_policy_version,
    )

    if dry_run:
        # §38: a dry run must answer "will I get a formal score, and why/why not".
        dry_run_info = {
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
        }
        if resolved_reference_set is not None:
            dry_run_info["Reference Calibration"] = "是 (auto)" if reference_set is None else "是"
            dry_run_info["ReferenceSet ID"] = reference_set_id or "N/A"
            dry_run_info["ReferenceSet Version"] = reference_set_version or "N/A"
            dry_run_info["ReferenceSet Content SHA"] = (
                (reference_set_content_sha256[:16] + "...") if reference_set_content_sha256 else "N/A"
            )
            dry_run_info["Calibration Policy"] = (
                f"{calibration_policy_id} {calibration_policy_version}" if calibration_policy_id else "N/A"
            )
        else:
            dry_run_info["Reference Calibration"] = "UNAVAILABLE"
            dry_run_info["说明"] = "未找到兼容的可信 ReferenceSet，不生成正式 0–100 能力分"
        dry_run_info["历史对比"] = "是" if compare_latest else "否"
        print_dry_run(dry_run_info)
        return

    api_key = check_api_key(config.api_key_env)
    if api_key is None:
        print_error(f"环境变量 {config.api_key_env} 不存在或为空", "配置检查", partial=False)
        raise typer.Exit(code=1)

    # §9: 自动发现唯一 ReferenceSet 时，在开始前明确显示校准信息。
    if resolved_reference_set is not None:
        _announce_reference_set(resolved_reference_set)

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
    # §Live / §Cancel: 协作取消 token + 进度渲染（TTY Rich Live / 非 TTY 阶段日志）。
    from llmtrace.execution.progress import CancellationToken

    cancel_token = CancellationToken()
    progress = CliProgress()
    try:
        runner = UnifiedAuditRunner(
            config,
            api_key=api_key,
            target_id=resolved_target,
            repository=repository,
            compare_latest=compare_latest,
            baseline_snapshot_path=baseline_snapshot,
            reference_snapshot_path=reference_snapshot,
            reference_set_path=resolved_reference_set,
            max_wall_seconds=max_wall_seconds,
            progress_sink=progress.on_event,
            cancel_token=cancel_token,
        )
    except SandboxUnavailableError as exc:
        print_error(describe_error(exc, api_key_env=config.api_key_env), "预检", partial=False)
        raise typer.Exit(code=1)

    # 第一次 Ctrl+C → 协作取消（完成当前项后安全退出）；第二次 → 强制退出 130。
    def _on_sigint(signum: int, frame: object) -> None:
        if cancel_token.cancelled:
            raise KeyboardInterrupt()
        cancel_token.cancel()
        progress.announce_cancel()

    previous_sigint = signal.signal(signal.SIGINT, _on_sigint)
    try:
        progress.start()
        try:
            result = asyncio.run(runner.run())
        except KeyboardInterrupt:
            raise typer.Exit(code=130)
        except Exception as exc:
            if debug:
                import traceback

                traceback.print_exc()
            scrubber = SecretScrubber([api_key, *extract_url_secret_values(config.base_url)])
            print_error(
                scrubber.scrub_text(describe_error(exc, api_key_env=config.api_key_env)), "统一执行", partial=True
            )
            raise typer.Exit(code=1)
    finally:
        progress.stop()
        signal.signal(signal.SIGINT, previous_sigint)

    artifact_paths = {
        "manifest.json": str(output_dir / "runs" / result.execution_id / "manifest.json"),
        "report.json": str(output_dir / "runs" / result.execution_id / "report.json"),
        "report.html": str(output_dir / "runs" / result.execution_id / "report.html"),
    }
    print_unified_summary(result, artifact_paths)


@app.command("doctor")
def doctor() -> None:
    """诊断本机环境：Python / 依赖 / 沙箱 / 参考集 / API Key（只读，不发送请求）."""
    import importlib.metadata
    import importlib.util
    import platform
    import sys

    from rich.console import Console as _RichConsole
    from rich.panel import Panel as _RichPanel
    from rich.table import Table as _RichTable

    _doctor_console = _RichConsole()
    # 关键依赖对 CLI 必需；Web 依赖标记为 optional。
    cli_deps = ("httpx", "rich", "pydantic", "typer")
    web_deps = ("fastapi", "uvicorn")
    failures: list[str] = []

    table = _RichTable(title="llmtrace doctor")
    table.add_column("检查项", style="cyan")
    table.add_column("值", style="white")
    table.add_column("状态", style="white")

    # 1) Python / 平台
    py_ok = sys.version_info >= (3, 10)
    table.add_row("Python", f"{platform.python_version()} ({platform.system()})", "OK" if py_ok else "FAIL")
    if not py_ok:
        failures.append("Python 版本过低（需 >= 3.10）")

    # 2) llmtrace 版本
    try:
        version = importlib.metadata.version("llmtrace")
        table.add_row("llmtrace", version, "OK")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - editable install 总有
        table.add_row("llmtrace", "未识别（非安装环境）", "WARN")
        failures.append("llmtrace 未以安装包形式存在（建议 pip install -e .）")

    # 3) 依赖
    for dep in cli_deps + web_deps:
        present = importlib.util.find_spec(dep) is not None
        required = dep in cli_deps
        status = "OK" if present else ("FAIL" if required else "WARN")
        table.add_row(dep, "已安装" if present else "缺失", status)
        if required and not present:
            failures.append(f"必需依赖缺失: {dep}")

    # 4) 代码沙箱（Docker）
    try:
        from llmtrace.adapters.code_execution import create_code_execution_backend

        backend = create_code_execution_backend()
        table.add_row("代码沙箱", type(backend).__name__, "OK")
    except Exception as exc:
        table.add_row("代码沙箱", "Docker 不可用", "WARN")
        _doctor_console.print(f"[yellow]  沙箱提示：{exc}[/]")
        _doctor_console.print(
            "  [yellow]影响：Quick Suite 中的编码类题目无法计分；协议审计与纯文本能力评估不受影响。[/]"
        )

    # 5) 参考集目录与可信集合数量（自动发现）
    try:
        from llmtrace.appdir import default_data_root
        from llmtrace.execution.artifacts import RunArtifactRepository
        from llmtrace.reference.discovery import discover_compatible_sets, reference_sets_dir

        sets_dir = reference_sets_dir()
        sets_dir.mkdir(parents=True, exist_ok=True)
        all_sets = sorted(p for p in sets_dir.glob("*.json"))
        trusted = discover_compatible_sets(RunArtifactRepository(default_data_root() / "reports"))
        status = "OK" if trusted else "WARN"
        table.add_row("参考集目录", str(sets_dir), status)
        detail = f"{len(all_sets)} 个文件 / {len(trusted)} 个可信 ReferenceSet"
        if trusted:
            detail += f"（{', '.join(p.stem for p in trusted[:3])}）"
        table.add_row("ReferenceSets", detail, status)
        if not trusted and not all_sets:
            pass  # 首次使用属正常，不列为 failure
    except Exception as exc:
        table.add_row("参考集目录", f"检查失败: {exc}", "WARN")

    # 6) 默认 API Key 环境变量
    from llmtrace.security.redaction import check_api_key

    has_default_key = check_api_key("LLMTRACE_API_KEY") is not None
    table.add_row("LLMTRACE_API_KEY", "已设置" if has_default_key else "未设置", "OK" if has_default_key else "WARN")

    _doctor_console.print()
    _doctor_console.print(_RichPanel.fit("llmtrace doctor", style="bold blue"))
    _doctor_console.print(table)
    if failures:
        _doctor_console.print("[bold red]发现需要修复的问题:[/]")
        for msg in failures:
            _doctor_console.print(f"  [red]- {msg}[/]")
        raise typer.Exit(code=1)
    _doctor_console.print("[green]环境基本就绪。下一步：llmtrace run --help 或参考 README Quick Start。[/]")


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


reference_app = typer.Typer(
    name="reference",
    help="Reference：捕获受信任参考快照并构建 ReferenceSet（v0.4-A）",
    no_args_is_help=True,
)
app.add_typer(reference_app, name="reference")


@reference_app.command("capture")
def reference_capture(
    protocol: str = typer.Option(DEFAULT_PROTOCOL, "--protocol", "-p", help="协议类型: openai 或 anthropic"),
    base_url: str = typer.Option(..., "--base-url", "-u", help="API Base URL"),
    model: str = typer.Option(..., "--model", "-m", help="声明模型名称"),
    api_key_env: str = typer.Option(
        DEFAULT_API_KEY_ENV, "--api-key-env", "-k", help="API Key 环境变量名（不直接传 key）"
    ),
    auth_style: str = typer.Option("auto", "--auth-style", help="鉴权方式: auto, bearer, x-api-key, both"),
    provider_id: str = typer.Option(..., "--provider-id", help="参考源操作标识（元数据，非身份声明）"),
    snapshot_id: str = typer.Option(..., "--snapshot-id", help="唯一 filename-safe ReferenceSnapshot 标识"),
    created_by: str = typer.Option(..., "--created-by", help="创建者标签（operator 或 tool）"),
    reference_dir: Path = typer.Option(
        Path("references"), "--reference-dir", help="reference 根目录（snapshots/ sets/）"
    ),
    output_dir: Path = typer.Option(
        Path("reference-runs"), "--output-dir", "-o", help="run artifact 根目录（runs/ 位于其下）"
    ),
    timeout: float = typer.Option(DEFAULT_TIMEOUT, "--timeout", "-t", help="请求超时(秒)"),
    max_wall_seconds: float = typer.Option(None, "--max-wall-seconds", help="整体墙钟超时(秒)"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="只显示执行计划：0 HTTP / 0 API Key / 0 artifact / 0 snapshot"
    ),
    non_interactive: bool = typer.Option(False, "--yes", "-y", help="跳过确认"),
    debug: bool = typer.Option(False, "--debug", help="显示完整异常堆栈"),
) -> None:
    """捕获一次受信任参考运行（复用 UnifiedAuditRunner，资格通过后保存 ReferenceSnapshot）.

    Operator 必须确认 endpoint 是可信参考源。LLMTrace 记录声明与测量 provenance，
    但不独立证明 endpoint 归属（§28）。
    """
    import sys

    from rich.console import Console as _RichConsole

    config = AuditConfig(
        protocol=Protocol(protocol),
        base_url=base_url,
        model=model,
        api_key_env=api_key_env,
        auth_style=AuthStyle(auth_style),
        repeat_count=DEFAULT_REPEAT_COUNT,
        timeout=timeout,
        check_streaming=True,
        output_dir=output_dir,
    )
    target = derive_target_id(config.protocol.value, config.base_url)
    service = ReferenceCaptureService(reference_dir=reference_dir, artifact_root=output_dir)
    plan = service.build_plan(config, target_id=target)

    if dry_run:
        print_dry_run(
            {
                "模型": config.model,
                "Provider": provider_id,
                "Suite": f"{plan.suite_id} {plan.suite_version}",
                "Suite Content SHA": plan.suite_content_sha256,
                "Generation Config SHA": plan.generation_config_sha256,
                "总请求上限": str(plan.maximum_requests),
                "输出 Token 上限": str(plan.maximum_output_token_ceiling),
                "需要安全 Sandbox": "是",
                "Snapshot ID": snapshot_id,
                "Reference 目录": str(reference_dir),
                "Artifact 目录": str(output_dir),
            }
        )
        return

    api_key = check_api_key(config.api_key_env)
    if api_key is None:
        print_error(f"环境变量 {config.api_key_env} 不存在或为空", "配置检查", partial=False)
        raise typer.Exit(code=1)

    if not non_interactive:
        if not sys.stdin.isatty():
            print_error("非交互式执行需要 --yes", "确认", partial=False)
            raise typer.Exit(code=1)
        print_dry_run(
            {
                "模型": config.model,
                "总请求上限": str(plan.maximum_requests),
                "Snapshot ID": snapshot_id,
            }
        )
        # Never echo a raw base URL: it may carry userinfo credentials or a
        # secret query parameter (§34).  redact_url is the single scrubber.
        if not typer.confirm(
            f"本次 reference capture 将向 {redact_url(config.base_url)} 发送最多 "
            f"{plan.maximum_requests} 个请求。是否继续？"
        ):
            raise typer.Exit(code=0)

    try:
        result = asyncio.run(
            service.capture(
                config=config,
                api_key=api_key,
                target_id=target,
                provider_id=provider_id,
                snapshot_id=snapshot_id,
                created_by=created_by,
                max_wall_seconds=max_wall_seconds,
            )
        )
    except KeyboardInterrupt:
        raise typer.Exit(code=130)
    except Exception as exc:
        if debug:
            import traceback

            traceback.print_exc()
        scrubber = SecretScrubber([api_key, *extract_url_secret_values(config.base_url)])
        print_error(
            scrubber.scrub_text(describe_error(exc, api_key_env=config.api_key_env)), "reference capture", partial=True
        )
        raise typer.Exit(code=1)

    if result.status == ReferenceCaptureStatus.CAPTURED:
        snapshot_path = reference_dir / "snapshots" / f"{result.snapshot_id}.json"
        sidecar_path = reference_dir / "snapshots" / f"{result.snapshot_id}.manifest.json"
        # §14: 完成输出要有明确下一步，不止是文件路径。
        _console = _RichConsole()
        _console.print()
        _console.print("[bold green]Reference capture complete.[/]")
        _console.print(f"Model:    [bold]{model}[/]")
        _console.print("Qualified: YES")
        _console.print(f"Snapshot: {snapshot_path}")
        _console.print()
        _console.print("[bold]Next:[/] 将本快照加入 ReferenceSet 以获得 0–100 正式校准：")
        _console.print(
            f"  [cyan]llmtrace reference set-create --set-id <id> --set-version 1 --snapshot {snapshot_id}[/]"
        )
        _console.print(f"  完整性锚点: {sidecar_path}")
    elif result.status == ReferenceCaptureStatus.QUALIFICATION_REJECTED:
        # §14: Qualification 失败明确打印 "Reference rejected" 及原因。
        _console = _RichConsole()
        _console.print()
        _console.print("[bold red]Reference rejected[/]")
        _console.print("Qualified: NO")
        _console.print(f"Reason: {list(result.reason_codes)}")
        _console.print("运行完成但未通过资格门禁，ReferenceSnapshot 未生成（run artifact 保留）。")
        _console.print(f"执行记录保留在: {output_dir / 'runs' / result.execution_id}")
        raise typer.Exit(code=1)
    else:
        print_error(f"参考运行失败: {result.warnings}", "reference capture", partial=True)
        typer.echo(f"     执行记录保留在: {output_dir / 'runs' / result.execution_id}")
        raise typer.Exit(code=1)


@reference_app.command("set-create")
def reference_set_create(
    reference_dir: Path = typer.Option(Path("references"), "--reference-dir", help="reference 根目录"),
    set_id: str = typer.Option(..., "--set-id", help="唯一 filename-safe ReferenceSet 标识"),
    set_version: str = typer.Option(..., "--set-version", help="set 修订版本（filename-safe）"),
    snapshots: list[str] = typer.Option(..., "--snapshot", help="成员 snapshot id（可重复传入）"),
    description: str = typer.Option("", "--description", help="可选描述"),
    debug: bool = typer.Option(False, "--debug", help="显示完整异常堆栈"),
) -> None:
    """从已验证的 ReferenceSnapshot 构建并保存 ReferenceSet（0 API 请求）.

    流程：load snapshot → verify trusted snapshot（sidecar 完整性锚点）→
    compatibility gate → ReferenceSetBuilder → ReferenceSetRepository.save（§31）。

    成员必须是 v0.4-A trusted snapshot（带 ``<snapshot_id>.manifest.json``
    完整性 sidecar）。v0.3-C legacy snapshot 没有锚点，仍可读取用于原始能力
    对比，但不能进入 trusted ReferenceSet。
    """
    from rich.console import Console as _RichConsole

    try:
        snapshot_repo = ReferenceRepository.load(reference_dir / "snapshots")
        set_repo = ReferenceSetRepository(directory=reference_dir / "sets")

        loaded = [snapshot_repo.get(sid) for sid in snapshots]
        sha_map = {sid: snapshot_repo.verify_trusted_snapshot(sid) for sid in snapshots}
        reference_set = ReferenceSetBuilder().build(
            reference_set_id=set_id,
            reference_set_version=set_version,
            created_at=datetime.now(UTC),
            snapshots=loaded,
            snapshot_sha256s=sha_map,
            description=description,
        )
        set_repo.save(reference_set)
    except (ReferenceNotFoundError, ReferenceError, ReferenceSetError) as exc:
        if debug:
            import traceback

            traceback.print_exc()
        print_error(str(exc), "reference set-create", partial=False)
        raise typer.Exit(code=1)

    set_path = reference_dir / "sets" / f"{set_id}_{set_version}.json"
    _console = _RichConsole()
    _console.print()
    _console.print("[bold green]ReferenceSet complete.[/]")
    _console.print(f"ID:       [bold]{set_id}[/] v{set_version}")
    _console.print(f"Members:  {len(reference_set.members)} trusted identities")
    _console.print(f"Set:      {set_path}")
    _console.print()
    _console.print("[bold]Next:[/] 直接运行审计即可自动发现本 ReferenceSet：")
    _console.print("  [cyan]llmtrace run --base-url <url> --model <model>[/]")


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host", help="监听地址（默认只绑定本机；--host 0.0.0.0 会打印安全警告）"),
    port: int = typer.Option(8765, "--port", min=1, max=65535, help="监听端口"),
    data_dir: str | None = typer.Option(None, "--data-dir", help="数据目录（默认 $LLMTRACE_HOME 或 ~/.llmtrace）"),
    demo: bool = typer.Option(
        False,
        "--demo",
        help="Demo 模式：进程内 Mock 模型端点 + 不安全 in-process sandbox（页面带 DEMO 标识）",
    ),
) -> None:
    """启动本地 Web 应用（v0.5 Usable MVP）。

    安全默认只绑定 127.0.0.1（页面会接收 API Key，§五十二）。API Key
    只存在于进程内存，SQLite / 报告 / 事件均不含 key（§二十二 / §六十）。
    """
    import uvicorn

    from llmtrace.appdir import ensure_app_layout
    from llmtrace.web.app import create_app

    layout = ensure_app_layout(Path(data_dir) if data_dir else None)

    if demo:
        typer.echo(
            "[WARN] --demo：代码项在进程内直接执行（UNSAFE），并使用内置 Mock 模型端点。"
            "仅用于演示，不要用来审计不可信模型的输出。"
        )

    if host in ("0.0.0.0", "::"):
        typer.echo(f"[WARN] 绑定 {host}：页面会接收 API Key，仅应在可信本地网络使用（§五十二）。")

    application = create_app(layout, demo=demo)
    typer.echo("LLMTrace Web")
    typer.echo(f"http://{host}:{port}")
    uvicorn.run(application, host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
