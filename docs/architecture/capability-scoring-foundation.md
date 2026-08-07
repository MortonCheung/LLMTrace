# Capability Scoring Foundation

## 概述

Capability Scoring 是 LLMTrace v0.3 的能力评分引擎底座。它从 Benchmark 客观评分结果出发，
通过能力维度归属、维度聚合和 Evidence 追溯，生成可审计的 `CapabilityProfile`。

## 关键区分：Objective Benchmark Score vs Calibrated Capability Score

```
Benchmark 客观评分              能力评分
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
normalized_score (0.0–1.0)       dimension_score (0.0–1.0)
从 GradeResult 得出                 从加权聚合得出
                                   ↓
                              calibrated_score (0–100) ← None (尚未)
```

LLMTrace 的最终产品定义：

- **0** = 无有效能力 / 无有效结果
- **50** = 可信参考模型中位水平
- **90** = 当前前沿参考水平
- **100** = 保留上限

但 `<–>` 以上的 `50`/`90` 含义依赖 **Reference Profile / Reference Model 数据库**，
该数据库尚未建立。

当前阶段 (`v0.3`) **只能**输出：

- `raw_normalized_score`：客观 Benchmark 聚合
- `provisional_raw_index`：已测维度加权和（不除以 coverage_weight）
- `calibrated_score` / `calibrated_total_score`：**None**

## 为什么不能直接输出正式 0–100

将 `normalized_score * 100` 伪装成最终能力分存在严重产品语义错误：

1. **未知测不到的上限**：`normalized_score=0.8` 在某个 Benchmark 上是什么能力水平？
   一个 13B 模型在 MMLU 上的 0.8 和一个 405B 模型在 MATH 上的 0.4 无法简单比较。

2. **跨模型可比性不足**：没有 Reference Profile，无法将"在任务集 A 上的 80% 准确率"
   转换为"相对当前前沿，该模型处于什么水平"。

3. **维度覆盖不完整**：当前只启用 4 个维度（reasoning, coding, math_science,
   instruction_following），覆盖了全局权重的 75%。未测维度不应被忽略或重新归一化。

## 为什么未测维度不能重新归一化

重新归一化会将不完整的测试结果伪装成完整能力评估：

```
示例：只测 reasoning + coding
  当前启用权重 = 0.25 + 0.20 = 0.45

  如果归一化：
    reasoning 贡献 = 0.8 × (0.25/0.45) = 0.8 × 0.556 = 0.444
    coding 贡献    = 0.7 × (0.20/0.45) = 0.7 × 0.444 = 0.311
    total = 0.755

  如果不归一化：
    reasoning 贡献 = 0.8 × 0.25 = 0.20
    coding 贡献    = 0.7 × 0.20 = 0.14
    total = 0.34

  归一化结果 0.755 会误导用户认为"模型总体能力 75.5%"
  实际只有 45% 权重被覆盖。
```

## coverage_weight 的意义

`coverage_weight` 表示当前有有效结果的维度在全局权重中的占比：

- 只测 reasoning (0.25) + coding (0.20) → `coverage_weight = 0.45`
- 全测 → `coverage_weight = 1.0`

用户可以通过 `coverage_weight` 理解评估的完整性，
也可以通过 `provisional_raw_index` 看到当前客观结果，
但不会被误导为"完整评估"。

## smoke 为什么永远不进入能力分

Smoke 任务是运行时的集成健康检查（比如验证 Provider 连接、任务加载正常等）。
其 `capability_score_eligible=False`，因此：

- 不参与维度分数计算
- 不参与 `provisional_raw_index`
- 不影响 `coverage_weight`
- 不影响 `task_coverage`

即使 smoke task 的 `normalized_score = 1.0`，也不会改变任何能力评分。

## Evidence 如何进入评分结果

Evidence 链从底层 API 调用一路向上：

```
HTTPEvidence
  → TaskAttempt.evidence_refs
    → GradeResult.evidence_refs
      → DimensionScoreResult.evidence_refs
        → CapabilityProfile.evidence_refs
```

聚合规则：

1. 只收集参与评分的任务的 Evidence（SUCCESS + GRADED + eligible）
2. 去重，保持 first-seen 顺序
3. 不创建或伪造 Evidence
4. UUID 格式与上游一致

## 后续 Reference Calibration 接入方式

`ScoreCalibrator` Protocol 定义了校准接口：

```python
class ScoreCalibrator(Protocol):
    def calibrate(
        self,
        dimension: CapabilityDimension,
        raw_score: float,
        reference_profile: object | None = None,
    ) -> float | None: ...
```

当前默认实现 `NoCalibration` 始终返回 `None`。

未来接入时：

1. 构建 Reference Profile 数据库（包含各维度参考模型的分数分布）
2. 实现 `ScoreCalibrator` 子类（如 `ReferenceModelCalibrator`）
3. 传入 `aggregate_capability_profile(calibrator=...)`
4. 所有维度状态从 `UNCALIBRATED` 变为 `SCORED`
5. `calibrated_score` 和 `calibrated_total_score` 获得真实值

## 当前阶段明确不做什么

- 正式 MMLU / GPQA / HumanEval 等大规模执行
- 下载大型数据集
- Reference Model 数据库
- 百分位 / Confidence Interval
- LLM-as-a-Judge
- Web 前端
- CLI 正式 evaluate 命令
- 正式 0–100 总分
- 模型排名
