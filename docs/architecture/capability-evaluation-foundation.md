# Capability Evaluation Foundation

## Overview

LLMTrace v0.2 建立统一的"评测粘合层"——在未来接入 lm-eval、LiveBench、EvalPlus、Inspect AI 等外部评测框架时，共同使用的数据模型和 Adapter 接口。

## Architecture

```
外部评测项目               LLMTrace 粘合层                  LLMTrace 基础设施
─────────────────────────────────────────────────────────────────────────
lm-eval ───┐
            │              BenchmarkAdapter (Protocol)
LiveBench ──┤                  │
            ├── Adapter ───────┼── Provider (统一请求出口)
EvalPlus ───┤                  │         │
            │              RunPlan           HTTPEvidence
Inspect AI ─┘              TaskAttempt ──────▶ Evidence UUID
                           GradeResult
                           BenchmarkRunResult
```

## Key Concepts

| 概念 | LLMTrace 角色 | 说明 |
|------|--------------|------|
| 外部评测项目 | 测量仪器 | lm-eval、LiveBench 等是黑盒测试工具 |
| BenchmarkAdapter | 插头 | 每个外部项目一个 Adapter，转换其格式 |
| Provider | 统一请求出口 | 所有 HTTP 请求通过 Provider，不做旁路 |
| HTTPEvidence | 证据源 | 评测过程产生的证据全部引用同一 Evidence ID 系统 |
| Benchmark 模型 | 统一结果格式 | TaskAttempt、GradeResult、BenchmarkRunResult 等 |

## Data Model

### Benchmark Source & Suite

- **BenchmarkSource** — 标识评测来源（如 `mmlu`、`livebench`）
- **SuiteVersion** — 不可变版本号（`model_config = {"frozen": True}`）
- **BenchmarkSuite** — 包含多个 TaskSpec 的评测套件

### Execution

- **TaskSpec** — 单个评测任务定义
- **RunPlan** — 确定性的执行计划（由 Planner 生成）
- **TaskAttempt** — 单次评测执行记录，引用 evidence UUIDs

### Results

- **GradeResult** — 评分结果，`normalized_score` 范围 [0, 1]
- **DimensionResult** — 单维度结果（准确率、F1 等）
- **BenchmarkRunResult** — 聚合运行结果

### Budget

- **BudgetEstimate** — 资源预估（请求数、Token 数、时长、费用）
- 价格不可用时 `estimated_cost` 为 `None`，不伪造金额

## Evidence Relationship

所有评测结果通过 `evidence_refs: list[str]` 引用现有 `HTTPEvidence.evidence_id`（UUID 字符串），不另建第二套证据系统。这与 `FindingResult.evidence_refs` 使用相同的设计模式。

## Adapter Protocol

`BenchmarkAdapter` 是抽象基类，定义 6 个必须实现的接口：

```
adapter_id          → str
adapter_version     → str
list_tasks()        → list[TaskSpec]
build_plan(...)     → RunPlan
estimate_budget(…)  → BudgetEstimate
run_task(...)       → TaskAttempt (async)
normalize_result(…) → GradeResult
```

### Design Constraints

- Adapter 不能直接读取 API Key
- Adapter 不能绕过 Provider
- Adapter 不能自建 HTTP 客户端
- 所有输出必须转换为统一的 TaskAttempt/GradeResult
- Adapter 失败必须返回结构化失败信息，不吞异常
- 设计允许未来 subprocess 隔离外部框架

## Registry

三个只读注册表：

- `BenchmarkSourceRegistry` — ID 唯一，重复注册抛 `DuplicateRegistrationError`
- `BenchmarkSuiteRegistry` — 同上
- `BenchmarkAdapterRegistry` — 同上

所有 registry 提供：`register()`、`get()`、`list_ids()`、`list_all()`。

## Planner

`build_plan()` 函数根据 Suite/TaskSpec 生成确定性的 RunPlan：

- 同一输入 → 完全相同的计划，包括相同的 `plan_id`
- `plan_id` 是规范化输入（suite/source/adapter 标识、结构化任务列表、预算与价格参数）的 SHA-256 摘要；任一关键参数变化都会产生不同的 `plan_id`
- 任务顺序是计划身份的一部分：`tasks` 参数的顺序影响 `plan_id`，语义为"任务列表顺序即执行顺序"
- 重试次数明确计入最大请求数：`maximum_requests = planned_requests * (1 + max_retries)`
- 未提供价格时 `estimated_cost = None`

## What Is NOT Included in This Round

- lm-evaluation-harness、LiveBench、EvalPlus、Inspect AI 接入
- 0–100 综合能力指数
- 参考模型库
- 行为指纹
- 混合路由
- SQLite / FastAPI
- 前端
- 真实 API 调用
- 修改现有 Provider 公共行为
- 插件自动发现
