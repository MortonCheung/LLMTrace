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

## v0.3-D Behavior Drift Foundation（已完成）

在**相同 Benchmark、相同题目、相同生成参数、相同评分规则**下，对同一目标 API 的两次运行
做行为漂移分析，并区分「能力结果变化 / 回答表现变化 / 运行状态变化」，把不可比数据 fail closed。

### 新增领域模型（`analysis/behavior_models.py`）

- `BehaviorItemKey`：跨 Run 稳定身份 `(task_id, source_sample_id, input_sha256)`，frozen；
  禁止用 `attempt_id`（每次运行不同）或 `item_id`（只表示运行内序号）
- `BehaviorItemObservation`：只存 hash / 长度 / operational 元数据，**不保存完整回答**
- `BehaviorRunSnapshot`：一次观测执行的不可变快照（≠ ReferenceSnapshot）
- `BehaviorDriftPolicy`：版本化阈值（`llmtrace_behavior_drift_v1` / 0.1.0，provisional）
- 输出规范化：CRLF/CR→LF + strip 首尾 whitespace + SHA-256（不 lowercase、不删内部空格）
- `generation_config_sha256`：canonical JSON（sort_keys + 稳定 separators）→ SHA-256

### Builder 与 Engine（`analysis/behavior_snapshot.py` / `behavior_drift.py`）

- `BehaviorSnapshotBuilder`：每 item 要求 `source_sample_id` + `input_sha256`；
  duplicate stable key 拒绝；evidence 引用必须**恰好一个**，缺失/歧义 fail closed；
  item 顺序 deterministic（按 stable key 排序）
- `BehaviorDriftEngine` Compatibility Gate（先于任何 delta，fail closed）：
  `suite_id → suite_version → source → adapter → scoring_policy → generation_config →
  stable item set → comparable coverage`
- 四类 Detector 插件化：Outcome / Status / Output / Operational，只产局部 `BehaviorSignal`
- 稳定对齐：以 `BehaviorItemKey` 建 map，两侧 item 顺序无关
- `BehaviorDriftLevel`：`NO_SIGNIFICANT_DRIFT / OBSERVED_DRIFT / MATERIAL_DRIFT / INCONCLUSIVE`

### 关键防伪 invariant

- FAILURE 的强制 0 分不进入 outcome delta——只算 status change，绝不伪装成能力下降
- Output hash 变化只算「行为变化观测」，不直接判 MATERIAL，更不是模型偷换证据
- 不可比较（suite / generation config / policy / item set / coverage 不一致）→ raise，不产出 Result
- 兼容性失败 ≠ INCONCLUSIVE（不可比较 ≠ 比较结果不确定）

### 报告

- JSON 新增 `behavior_drift` 段，`SCHEMA_VERSION` 1.1 → 1.2
- HTML 新增 Behavior Drift 区域（Drift Level / Outcome / Status / Output / Graded Overlap /
  Dimension / Changed Items），保持 Jinja autoescape，不展示完整 output hash
- 顺手修复旧 `analysis/drift.py` 计数 bug：5 个 drift signal 不再错误退回 INCONCLUSIVE

### 开源设计参考

`docs/architecture/open_source_influences.md` 记录对 Nuclei / garak / Promptfoo / Evidently
的架构思想借鉴（Detector 解耦、插件化、versioned policy），本轮未复制任何源码、未新增 runtime 依赖。

- 质量：Ruff + Format + Mypy + Pytest 全量回归

完成标准：同一 Target API 两次可比 Quick Suite 运行 → 两个不可变 BehaviorRunSnapshot →
稳定身份逐项对齐 → 插件化 Detector 分离 Outcome/Status/Output/Operational →
版本化 Policy 得出保守 Drift Result，不可比时 fail closed，结论可回溯 Evidence。√ 已达成。

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
