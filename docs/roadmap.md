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

## v0.2 评测粘合层与能力评分基础（已完成）

- Source / Suite / Task / Run / Score 统一数据模型
- Benchmark Adapter Protocol（`list_tasks`、`run_task`、`normalize_result`）
- RunPlan 与 Budget Estimator
- `lm-evaluation-harness` 最小适配器（subprocess 隔离、YAML 任务加载）
- 统一结果模型与 Evidence 集成
- 真实上游 Benchmark 验收：GSM8K 8 样本固定子集（generate_until，Mock Provider）
- 能力维度评分引擎：推理 / 编程 / 数学与科学 / 指令遵循（4 个启用维度）
- `TaskScoringRegistry`：显式 `task_id → dimension` 映射
- `DimensionScoreResult` + `CapabilityProfile`（uncalibrated，无 0–100 输出）
- Evidence 闭合链：`HTTPEvidence → TaskAttempt → GradeResult → DimensionScore → CapabilityProfile → Report`
- Cross-run isolation：per-run GradeResult 配对，TaskAttempt 预扫描 duplicate 检测

完成标准：一个外部任务通过 Mock Provider 跑通完整闭环并进入 Evidence 与报告。√ 已达成。

## v0.3 能力评分 MVP（规划中）

> 注：Reference Calibration 是正式 0–100 能力分的必要条件。当前 `calibrated_score = None`。

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
