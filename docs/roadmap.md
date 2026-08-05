# LLMTrace 路线图

路线图与 [docs/product/PRD.md](./product/PRD.md) 保持同步，每个版本对应 PRD「开发里程碑」中的一个阶段。

## v0.1 证据审计 MVP（已完成）

- 协议证据采集（OpenAI-compatible / Anthropic-compatible，Base URL 与鉴权适配）
- 基础探针：连接、鉴权、模型列表、基线、无效模型、流式、元数据、稳定性
- 模型标识漂移检测与风险分析（LOW / MEDIUM / HIGH / INCONCLUSIVE）
- Evidence 唯一 ID 与 Finding 引用，原始响应 SHA-256、Token、延迟记录
- JSON / HTML 报告与跨报告比较
- honest / fallback / inconsistent Mock Server
- CLI：`audit`、`inspect`、`compare`
- 质量基线：GitHub Actions CI（Python 3.11 / 3.12 / 3.13、Ruff、Format、Mypy、Pytest、覆盖率门禁）

## v0.2 评测粘合层与能力评分基础（开发中）

- Source / Suite / Task / Run / Score 统一数据模型
- Benchmark Adapter Protocol（`list_tasks`、`build_plan`、`estimate_cost`、`run_task`、`normalize_result`）
- RunPlan 与 Budget Estimator
- 首个 `lm-evaluation-harness` 最小适配器（subprocess 隔离、固定 revision）
- 统一结果模型与 Evidence 集成

完成标准：一个外部任务通过 Mock Provider 跑通完整闭环并进入 Evidence 与报告。

## v0.3 能力评分 MVP

- 接入外部评测：lm-eval 精选任务、LiveBench 精选客观题、EvalPlus 精选代码题
- 快速模式，20–40 个任务
- 4 个能力维度：推理、数学、代码、指令遵循
- 0–100 分综合评分与置信度
- 成本估算与 JSON / HTML 报告

## v0.4 参考模型对照

- ReferenceProfile 参考模型体系
- 官方 API 实测基线数据
- 声明模型差距与分项差距
- 版本化参考组

## v0.5 行为相似度与混合路由

- 主动指纹题与行为向量
- Top-K 相似度
- Model Equality Testing 统计模式
- 混合路由与跨时间动态降级

## v0.6 简易 Web 前端

- 输入接口与成本确认
- 开始测试与实时进度
- 结果报告与历史记录
- 不依赖外部服务，可本地运行
