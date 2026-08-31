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
- 不输出身份结论（只说 lower/higher），无 0–100 评分，无 Calibration
- JSON `reference_comparison` 段 + HTML Reference Comparison 区域，保持 UNCALIBRATED 警告

### 收口后的基础 invariant

- **Append-only 在磁盘上生效，不只是内存**：`save()` 使用 exclusive-create（`open("x")`）写盘，`FileExistsError` 转为 `DuplicateSnapshotError`；先写盘成功、再登记内存索引，写盘失败不污染内存
- **`snapshot_id` 即安全文件名 stem**：`^[A-Za-z0-9][A-Za-z0-9._-]*$`，拒绝路径分隔与 `..`；`_file_path()` 另有 containment 二次校验（解析后必须是仓库目录的直接子文件）
- **`suite_sha256` 是真 SHA-256**：64 位 hex 校验，进入模型后统一 lowercase
- **`created_at` 必须 timezone-aware**：`ReferenceSnapshot` 与 `ReferenceProvenance` 均拒绝 naive datetime，并 normalize 到 UTC
- **Coverage 按 `DimensionScoreStatus` 判定**：`SCORED` / `UNCALIBRATED` 可比，`UNAVAILABLE` / `INSUFFICIENT_DATA` 视为未测量，绝不折算成 0 分参与 delta
- **Compatibility Gate 先于 delta**：`suite_id` → `suite_version` → `scoring_policy_id` → `scoring_policy_version` → 可比维度集合 → `coverage_weight` → 逐维 delta，任何一步失败 fail closed，不产生 `ComparisonResult`
- **Scoring policy 是独立概念**：新增 `ScoringPolicyMismatchError`（`SCORING_POLICY_MISMATCH`），不复用 `SuiteMismatchError`

### 语义边界

```text
Reference Snapshot      = 不可变的历史能力事实
Repository              = append-only 持久化
Comparator              = compatibility gate + 相对能力 delta
Reference Comparison   != Calibration
Reference Comparison   != 模型身份识别
部分覆盖                != 能力下降
不同 scoring policy     != 可直接比较的画像
```

- 质量：Ruff + Format + Mypy + Pytest 836 题 + Coverage 83%

完成标准：第一套 Reference Model Snapshot 与能力比较基础设施落地，且历史事实不可覆盖、部分覆盖不伪装成能力下降。√ 已达成。

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
