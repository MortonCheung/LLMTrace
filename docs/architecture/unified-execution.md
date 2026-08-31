# Unified Execution & Artifact Foundation

`llmtrace run` 是 LLMTrace 的一键真实审计执行链：从 API 地址、Key 环境变量和声明模型出发，
完整跑通 协议审计 → Quick Suite 32 题 → 能力画像 → 行为快照 →（可选）历史漂移 / 参考对比 →
统一报告 → append-only 本地工件。

本文件记录这层执行链的架构约定与语义边界。领域模型见 `src/llmtrace/execution/`。

## 执行生命周期

```text
PRECHECK ──> PLAN ──> PROTOCOL ──> BENCHMARK ──> SCORING ──> SNAPSHOT
   └──> HISTORY COMPARISON ──> REFERENCE COMPARISON ──> REPORT/ARTIFACT COMMIT
```

每个阶段职责单一、可独立测试：

- **PRECHECK**：配置校验、API key 环境变量校验、Quick Suite 资源完整性、Docker sandbox
  可用性、reference/baseline 文件可解析、artifact 输出路径可写。任何一步失败在**发送任何
  API 请求之前**中止（fail closed）。
- **PLAN**：在请求前构造 `UnifiedExecutionPlan`（协议探针请求数 + 32 题 benchmark 请求数 +
  总请求硬上限 + 最大输出 token ceiling + generation_config_sha256）。
- **PROTOCOL**：`ProtocolAuditExecutor` 跑八类协议探针；config precheck / connectivity 失败
  是 **blocking failure**，此时跳过 benchmark，避免浪费 32 次调用。
- **BENCHMARK / SCORING / SNAPSHOT**：`QuickSuiteRunner` 跑 32 题 → `aggregate_capability_profile`
  → `BehaviorSnapshotBuilder` 构造行为快照。
- **HISTORY / REFERENCE COMPARISON**：可选。兼容性由 `BehaviorDriftEngine` / `CapabilityComparator`
  的 fail-closed gate 判定。
- **REPORT / ARTIFACT COMMIT**：统一 JSON/HTML 报告 + append-only 工件目录。

## Central Evidence Recorder

Provider 层负责对每个真实 HTTP 请求 **exactly-once** 记录：

- `EvidenceRecorder` 协议只有一个 `record(evidence)`；`InMemoryEvidenceRecorder` 按到达顺序
  保存，重复 `evidence_id` fail closed（不 last-write-wins）。
- `BaseProvider` 可选接受 recorder：`complete` / `list_models` / `stream_complete` 在
  成功、HTTP 4xx/5xx、异常三种路径上各记录一次。
- `evidence_recorder=None` 时保持旧行为——旧 `llmtrace audit` 与协议探针不受影响。
- 统一报告中 `Protocol Evidence + Benchmark Evidence = 完整 Run Evidence`。Quick Suite 通过
  Provider 拿到 `evidence_id`，真实 `HTTPEvidence` 对象由 recorder 统一交回 orchestration，
  供 `BehaviorSnapshotBuilder` 与报告共用同一份证据源。

## Request Budget

`RequestBudget` 是"一次审计总请求硬上限"的唯一执法点：

- Provider 在真正发送请求**之前** `consume(1)`；超出即 `RequestBudgetExceededError`，
  请求永不离开进程。
- 失败的 HTTP 请求同样消耗 budget——它确实被发送了。
- 预算来自 `UnifiedExecutionPlan.maximum_requests`，不是 CLI 显示的数字；两者必须一致。

## Quick Suite Generation Invariants

Quick Suite 是固定测量尺：`temperature = 0.0`、`max_tokens = 512`，且**真正发送的
generation config = BehaviorRunSnapshot 的 hash 输入 = Execution Plan 声明的 config**。

三者都来自唯一 canonical source：`QUICK_SUITE_GENERATION_CONFIG` +
`get_quick_suite_generation_config()`，不允许在 adapter / snapshot caller / planner 里各写一份字面量。

CLI 不暴露 `--temperature` 等 override——一旦允许用户改生成参数，历史 Drift 立即不可比。
未来若支持，必须进入 `generation_config_sha256` 并作为新的测量条件。

## Artifact Repository

`RunArtifactRepository` 是 append-only 文件系统 store，一次执行 = 一个不可变目录：

```text
reports/runs/<execution_id>/
    manifest.json           (最后写入)
    report.json
    report.html
    capability_profile.json
    behavior_snapshot.json
    benchmark_runs.json
```

- `execution_id` 由 UUID 生成，用户输入绝不直接成为目录名。
- Commit 流程：写 staging 目录 → 写全部 artifact → 计算 SHA-256 → 写 manifest（最后）→
  `rename` 原子落位。中途失败清理 staging，不留"看起来完成但少文件"的 run。
- 同 `execution_id` 覆盖历史目录 → `DuplicateExecutionError`。
- manifest 只含脱敏 target 元数据与 artifact hash，绝无 API key / Authorization / raw headers。

## History Baseline Selection

- `llmtrace run` 默认 `--compare-latest`：在提交当前 run 之前查找同一 target/model 的历史
  `BehaviorRunSnapshot`，newest → oldest，逐份走 `BehaviorDriftEngine` compatibility gate，
  取第一份真正 compatible 的做比较。
- **Repository 只做候选筛选**（target_id + candidate_model_id），最终兼容性权威始终是
  `BehaviorDriftEngine.compare()`。
- **Current run 绝不与自身比较**：`find_behavior_snapshots` 显式排除当前 `execution_id`。
- 不兼容候选 → skip 并记录 diagnostic warning；都不兼容 → 无 drift result，**禁止降低 gate**。
- 首次运行无历史 → 正常状态（`behavior_drift = None`），不降级 run status。

## Failure Boundaries

- **Protocol blocking failure**（401/403/unreachable）→ 停止 benchmark、保存 PARTIAL 工件、
  `CapabilityProfile = None`、`BehaviorSnapshot = None`，不制造 0 分画像。
- **Per-item provider failure**（如 3/32 timeout）→ 沿用 Quick Suite fixed-denominator 语义：
  item FAILURE + Evidence；后续 Behavior Drift 已能区分 provider failure 与真实能力漂移。
- **Internal integrity failure**（duplicate Evidence UUID / provenance mismatch / snapshot closure
  失败 / duplicate stable item key）→ fail closed，不降级成 warning 继续出"正式结论"。
- **Wall-clock timeout** → `asyncio.timeout` 包裹，provider 关闭、标记 FAILED、不伪装 success。

## Security

- API Key 只存在于 `os.environ → 内存 → Provider`，绝不写盘、写日志、写报告、写异常堆栈。
- Base URL 经 `redact_url` 脱敏（userinfo 凭据 + 敏感 query 值）后才进入 report / manifest /
  console。
- HumanEval 生产路径唯一允许 `DockerCodeExecutionBackend`；Docker 不可用 → preflight fail、
  0 benchmark 请求。**不存在 silent unsafe in-process fallback**——`_InProcessExecutionBackend`
  仅在单元测试显式注入时可达。

## 与其他概念的关系

- `llmtrace audit` = protocol-only legacy/advanced 命令（复用了同一个 `ProtocolAuditExecutor`）。
- `llmtrace run` = 推荐的一键统一审计。
- `llmtrace compare` = 旧协议/运维报告漂移比较，语义**不变**，不与 Behavior Drift 混为一谈。
- `llmtrace inspect` 向后兼容，旧报告无新段时正常显示。
