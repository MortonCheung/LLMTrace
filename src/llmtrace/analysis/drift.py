"""跨报告漂移分析."""

from __future__ import annotations

import statistics
from typing import Any

from llmtrace.models.report import CompareResult, DriftLevel


def compare_reports(reports_data: list[dict[str, Any]]) -> CompareResult:
    """比较多份报告."""
    if len(reports_data) < 2:
        result = CompareResult(
            reports=[r.get("report_id", "unknown") for r in reports_data],
            report_count=len(reports_data),
            report_times=[],
            endpoints=[],
            claimed_models=[],
            test_suite_versions=[],
        )
        result.warnings.append("至少需要两份报告进行比较")
        return result

    result = CompareResult(
        reports=[r.get("report_id", "unknown") for r in reports_data],
        report_count=len(reports_data),
        report_times=[r.get("meta", {}).get("utc_time", "") for r in reports_data],
        endpoints=[r.get("config", {}).get("base_url", "") for r in reports_data],
        claimed_models=[r.get("config", {}).get("model", "") for r in reports_data],
        test_suite_versions=[r.get("meta", {}).get("test_suite_version", "") for r in reports_data],
    )

    # 检查版本一致性
    versions = set(result.test_suite_versions)
    if len(versions) > 1:
        result.version_mismatch = True
        result.warnings.append(f"测试套件版本不一致: {sorted(versions)}，部分指标不可直接比较")

    # 提取成功率
    for r in reports_data:
        evidence = r.get("evidence", [])
        if evidence:
            successes = sum(1 for e in evidence if e.get("success", False))
            result.success_rates.append(successes / len(evidence))
        else:
            result.success_rates.append(0.0)

    # 提取延迟
    for r in reports_data:
        evidence = r.get("evidence", [])
        latencies = [e.get("total_latency_ms", 0) or 0 for e in evidence if e.get("success", False)]
        if len(latencies) >= 2:
            result.latency_medians_ms.append(statistics.median(latencies))
            mad = statistics.median(abs(x - result.latency_medians_ms[-1]) for x in latencies)
            result.latency_mads_ms.append(mad)
        elif latencies:
            result.latency_medians_ms.append(latencies[0])
            result.latency_mads_ms.append(0.0)

    # 提取返回模型集合
    for r in reports_data:
        evidence = r.get("evidence", [])
        models = {e.get("response_model", "") for e in evidence if e.get("response_model")}
        result.response_model_sets.append(sorted(models))

    # 提取结构指纹
    for r in reports_data:
        result.fingerprint_sets.append(r.get("schema_fingerprints", []))

    # 提取 Token 字段存在率
    for r in reports_data:
        evidence = r.get("evidence", [])
        if evidence:
            token_present = sum(
                1 for e in evidence if e.get("input_tokens") is not None and e.get("output_tokens") is not None
            )
            result.token_field_rates.append(token_present / len(evidence))
        else:
            result.token_field_rates.append(0.0)

    # 提取请求 ID 存在率
    for r in reports_data:
        evidence = r.get("evidence", [])
        if evidence:
            rid_present = sum(1 for e in evidence if e.get("response_id") is not None)
            result.request_id_rates.append(rid_present / len(evidence))
        else:
            result.request_id_rates.append(0.0)

    # 提取错误类型
    for r in reports_data:
        evidence = r.get("evidence", [])
        errors = {
            e.get("exception_type") or f"HTTP_{e.get('http_status')}" for e in evidence if not e.get("success", False)
        }
        result.error_types.append(sorted(errors))

    # 提取风险等级
    result.risk_levels = [r.get("risk_level", "INCONCLUSIVE") for r in reports_data]

    # 判断漂移程度
    result.drift_level, result.drift_notes = _compute_drift(result)

    return result


def _compute_drift(result: CompareResult) -> tuple[DriftLevel, list[str]]:
    """计算漂移程度."""
    notes: list[str] = []
    drift_signals = 0

    # 检查成功率变化
    if len(result.success_rates) >= 2 and max(result.success_rates) - min(result.success_rates) > 0.3:
        drift_signals += 1
        notes.append("成功率出现显著变化")

    # 检查延迟变化
    if len(result.latency_medians_ms) >= 2 and max(result.latency_medians_ms) > min(result.latency_medians_ms) * 2:
        drift_signals += 1
        notes.append("延迟中位数出现显著变化")

    # 检查返回模型集合变化
    if len(result.response_model_sets) >= 2:
        all_models = set()
        for s in result.response_model_sets:
            all_models.update(s)
        if len(all_models) > max(len(s) for s in result.response_model_sets):
            drift_signals += 1
            notes.append("返回模型集合发生变化")

    # 检查指纹变化
    if len(result.fingerprint_sets) >= 2:
        all_fps = set()
        for s in result.fingerprint_sets:
            all_fps.update(s)
        if len(all_fps) > max(len(s) for s in result.fingerprint_sets):
            drift_signals += 1
            notes.append("响应结构指纹发生变化")

    # 检查风险等级变化
    if len(set(result.risk_levels)) > 1:
        drift_signals += 1
        notes.append("风险等级发生变化")

    # NOTE: 5 signals is the maximum this function can produce (success rate,
    # latency, model set, fingerprint, risk level).  A high signal count must
    # escalate to LIKELY_DRIFT, never regress to INCONCLUSIVE — INCONCLUSIVE is
    # reserved for data that cannot be compared, not for "too many changes".
    if drift_signals == 0:
        return DriftLevel.NO_SIGNIFICANT_DRIFT, notes
    elif drift_signals <= 2:
        return DriftLevel.POSSIBLE_DRIFT, notes
    else:
        return DriftLevel.LIKELY_DRIFT, notes
