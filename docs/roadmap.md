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

## v0.3-A Item-Level Benchmark Foundation（已完成）

- `BenchmarkItemResult` 逐题评分与独立 Evidence 引用
- Item 级别 identity（`item_id`、`source_sample_id`、`input_sha256`）
- `GradeResult` 完整状态模型（GRADED / UNGRADABLE / FAILURE）
- `TaskAttempt.item_results` = canonical per-item truth
- 固定 denominator 不变：`sum(graded normalized_score) / planned_item_count`
- `normalize_result()` 强制验证 `planned_item_count` 与 `item_results` 长度一致性
- `aggregate_item_results()` 统一聚合逻辑（grading_coverage、execution_coverage）
- CodeExecutionBackend 抽象（Docker sandbox + InProcess fallback）

完成标准：单一 GSM8K 8 样本子集跑通完整 item-level 闭环。√ 已达成。

## v0.3-B Quick Suite（已完成）

- 四维 32 题 Quick Suite：`llmtrace_quick_v1` v0.1.0
  - `arc_challenge_quick_v1` → reasoning（8 题多选）
  - `humaneval_quick_v1` → coding（8 题 pass@1，Docker sandbox）
  - `gsm8k_quick_v1` → math_science（8 题 numeric exact match）
  - `ifeval_quick_v1` → instruction_following（8 题 atomic constraint）
- Immutable fixed subsets：SHA-256 排名算法，不可手工 cherry-pick
- 逐题 provenance 追踪（`source_sample_id`、`input_sha256`）
- TaskScoringRegistry 显式映射：`create_quick_registry()` 工厂函数
- CapabilityProfile：`coverage_weight = 0.75`，`provisional_raw_index` 不重归一化
- 所有 `calibrated_score` 保持 `None`
- Docker 安全执行（`--network none`、非 root、read-only rootfs）
- Provider failure isolation（单题失败不终止整套 32 题）
- Mock Provider 全 pipeline acceptance tests + 30+ adversarial tests
- JSON / HTML 报告 Quick Suite 汇总、UNCALIBRATED 警告
- 质量：Ruff + Format + Mypy + Pytest 728 题 + Coverage 82.5%

完成标准：四维全部 8/8 满分闭环 `provisional_raw_index = 0.75`。√ 已达成。

## v0.3-C Reference Model Snapshot（已完成）

- `ReferenceSnapshot`：不可变、追加式参考模型能力画像（`snapshot_id` / `model_id` / `provider_id` / `suite_id` / `suite_version` / `capability_profile` / `provenance`）
- `ReferenceProvenance`：可审计溯源（`source_type` / `created_by` / `created_at` / `suite_sha256` / `benchmark_revision` / `runner_version`）
- `ReferenceRepository`：JSON fixture 存储（`save` / `get` / `list` / `find_by_model`），重复 `snapshot_id` 拒绝
- `CapabilityComparator` + `ComparisonResult` + `DimensionDiff`：参考画像 vs 候选画像逐维度比较（`delta = candidate − reference`）
- 强制一致：suite_id / suite_version / 维度覆盖不一致分别抛 `SuiteMismatchError` / `SuiteVersionMismatchError` / `IncompatibleCoverageError`
- 不输出身份结论（只说 lower/higher），无 0–100 评分，无 Calibration
- JSON `reference_comparison` 段 + HTML Reference Comparison 区域，保持 UNCALIBRATED 警告
- 质量：Ruff + Format + Mypy + Pytest + Coverage 门禁通过

完成标准：第一套 Reference Model Snapshot 与能力比较基础设施落地。√ 已达成。

## v0.3-D Behavior Drift Detection（规划中）

> 注：v0.3-C 明确不包含 Calibration 与模型身份识别。正式 0–100 能力分与 Calibration 不在本阶段。

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
