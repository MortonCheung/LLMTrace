# LLMTrace - 模型寻迹

面向第三方 AI API、中转站和代理服务的黑盒模型审计工具。

## 当前版本能做什么

当前已完成 v0.1 证据审计 MVP、v0.2 基准评测基础设施、v0.3 能力评测收口
（item-level 结果、Quick Suite 32 题、Reference Model Snapshot、Behavior Drift Foundation），
以及 v0.3-E 一键统一审计执行链（`llmtrace run`）。

### v0.1 证据审计（基础能力）

- 统一执行链：每次请求只发送一次，探针直接返回真实证据（Evidence）与发现（Finding），CLI 仅负责编排汇总
- 每条证据带唯一 `evidence_id`（UUID），所有 Finding 的 `evidence_refs` 引用真实存在的证据 ID
- 证据按类型分类（baseline / streaming_comparison / streaming_baseline / invalid_model / model_catalog / connectivity），正常基线与流式对照属于不同证据类型；稳定性与元数据分析仅针对基线证据
- 无效模型探针使用随机假模型名请求，证据中同时记录配置声明模型、本次随机无效模型和服务端返回模型
- 模型标识漂移（同一会话返回多个不相关模型标识）独立形成 HIGH 风险
- 响应哈希基于完整原始响应字节计算（SHA-256），摘要再按字节截断，截断时标记 `truncated=true`；`response_body_size` 单位是字节
- 流式首 Token 延迟从首段非空文本 delta 计算，记录流开始/首文本/结束时间
- 请求次数以统一 AuditPlan 为准：dry-run 计划次数与实际执行完全一致；报告证据数与实际 HTTP 请求计数通过 Mock Server 调试接口独立核验
- 多报告跨时间漂移比较
- 密钥自动脱敏，不写入报告

### v0.2 基准评测基础设施（新增）

- **lm-evaluation-harness Adapter**：通过 `ProviderBackedLM` 桥接 LLMTrace Provider，支持 generate_until 任务
- **GSM8K 验收切片**：8 样本固定子集，可溯源至 openai/gsm8k，固定顺序与 ID，可复现
- **Benchmark 报告**：JSON/HTML 报告、Evidence closure validation、provenance 完整性
- **能力评分基础**：TaskScoringRegistry → Dimension Aggregation → CapabilityProfile
- **当前评分边界**：仅输出 raw_normalized_score / dimension_score / provisional_raw_index；calibrated_score 为 None（正式 0-100 需要 Reference Calibration）

### v0.3 能力评测收口（新增）

- **Item-level 结果**：`BenchmarkItemResult` 逐题评分、固定 denominator、Evidence 逐题追溯
- **Quick Suite**：四维 32 题固定子集（ARC / HumanEval / GSM8K / IFEval 各 8 题，SHA-256 排名不可 cherry-pick）
- **Reference Model Snapshot**：不可变、append-only 的参考模型能力画像；`CapabilityComparator` 带 fail-closed Compatibility Gate
- **Behavior Drift Foundation**：对同一 Target API 两次可比运行做行为漂移分析，区分能力结果 / 回答表现 / 运行状态变化，不可比数据 fail closed

### v0.3-E 统一审计执行链（新增）

- **`llmtrace run`**：一键真实审计——只提供 API 地址、Key 环境变量、声明模型，自动跑通
  协议审计 → Quick Suite 32 题 → CapabilityProfile → BehaviorRunSnapshot →（可选）历史漂移 / 参考对比 →
  统一 JSON/HTML 报告 → append-only 本地工件
- **真实 model 传播**：Quick Suite 使用用户声明的 model，不再硬编码 `test-model`
- **安全 sandbox fail closed**：HumanEval 生产路径唯一允许 Docker sandbox，Docker 不可用则 preflight 失败、0 次 benchmark 请求
- **中央 Evidence Recorder + Request Budget**：每个真实 HTTP 请求 exactly-once 记录，总请求数有硬上限
- **append-only 工件库**：一次执行 = 一个不可变目录（manifest + 报告 + 能力画像 + 行为快照 + benchmark 结果），含 SHA-256 校验

### v0.4-A 受信任参考运行与 ReferenceSet 基础（实验性，新增）

- **Suite Content Identity**：Quick Suite manifest 的 canonical content SHA-256（`suite_content_sha256`），
  贯穿执行计划、运行工件与参考快照 provenance
- **Reference Qualification（Gate 1–10）**：对一次落盘的运行工件逐道门禁 fail closed 校验，
  只有 32/32 全部评分的可信运行才能成为 ReferenceSnapshot，机器可读 reason codes
- **ReferenceSnapshotBuilder**：verify → qualify → build → save，provenance 记录 execution_id、
  adapter、generation config、run manifest 与 capability profile 的 SHA、qualification policy 与
  benchmark revisions
- **ReferenceSet / ReferenceSetRepository**：12-gate Compatibility 校验、canonical content hash 自校验、
  append-only 磁盘存储；生产 builder 拒绝 `test_fixture` 来源
- **CLI**：`llmtrace reference capture`（Operator 对可信 endpoint 捕获参考运行）与
  `llmtrace reference set-create`（从已验证快照构建参考组，0 API 请求）

> 注意：`ReferenceSet` 当前**不产生 calibrated 0–100 分数**。参考快照/参考组是可信、
> 不可变、可审计的历史能力事实；正式 0–100 能力分属于 v0.4-B Calibration（规划中）。

## 语义边界：LLMTrace 能说什么、不能说什么

- ✅ 能说：「在相同测试条件下，本次运行与历史运行观察到显著行为漂移。」
- ❌ 不能说：「模型被确定偷换成了 XXX。」
- ✅ 能说：「输出行为发生变化。」
- ❌ 不能说：「输出文本不同，因此底层模型不同。」
- ✅ 能说：「能力结果显著下降。」
- ❌ 但如果主要原因是 Provider Failure，不能说：「模型能力显著下降。」

## 当前版本不能证明什么

- 不能识别中转站真实上游模型（当前版本不具备模型指纹识别能力）
- 不能证明服务商是否使用了声明模型
- LOW 风险等级只表示"本次有限测试未发现明显异常，不代表已证明真实上游模型身份"
- 单次延迟不能直接判断模型身份
- 输出文本不同不代表底层模型不同；行为相似度不是身份结论
- Provider 失败（超时/限流）不等价于模型能力下降

## 安装

```bash
git clone https://github.com/MortonCheung/LLMTrace.git
cd LLMTrace
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

要求 Python 3.11+。

## 使用示例

### 一键统一审计（推荐）

```bash
export MY_API_KEY="your-key"
llmtrace run \
  --protocol openai \
  --base-url https://api.example.com/v1 \
  --model claimed-model \
  --api-key-env MY_API_KEY
```

一次 `run` 大约包含：协议探针（若干）+ Quick Suite 32 题 benchmark。实际请求数以
`--dry-run` 为准；运行前会显示预计请求数、最大输出 token ceiling 与预计费用（未知），
确认后才执行。`--yes` 跳过确认；`--compare-latest`（默认）自动与最新兼容历史运行做 Behavior Drift。

```bash
# 只显示执行计划，不发送任何请求（0 HTTP、0 工件、不要求 API key 存在）
llmtrace run --protocol openai --base-url https://api.example.com/v1 --model demo --api-key-env MY_API_KEY --dry-run
```

### OpenAI-compatible 接口审计（protocol-only，legacy/advanced）

```bash
export OPENAI_API_KEY="your-key"
llmtrace audit \
  --protocol openai \
  --base-url https://api.example.com \
  --model gpt-4 \
  --api-key-env OPENAI_API_KEY
```

### Anthropic-compatible 接口审计

```bash
export ANTHROPIC_API_KEY="your-key"
llmtrace audit \
  --protocol anthropic \
  --base-url https://api.example.com \
  --model claude-3-opus-20240229 \
  --api-key-env ANTHROPIC_API_KEY \
  --auth-style x-api-key
```

### Dry-run（不发送请求，仅显示执行计划）

```bash
llmtrace audit --protocol openai --base-url https://api.example.com --model test --api-key-env OPENAI_API_KEY --dry-run
```

### 查看报告

```bash
llmtrace inspect reports/llmtrace_20260804_120000_abc123.json
```

### 比较报告

```bash
llmtrace compare reports/run_a.json reports/run_b.json
```

### Reference workflow（实验性，v0.4-A）

先 `--dry-run` 查看计划（0 HTTP、0 工件、不要求 API key 存在）：

```bash
llmtrace reference capture \
  --protocol openai \
  --base-url https://api.example.com/v1 \
  --model reference-model \
  --api-key-env MY_API_KEY \
  --provider-id operator \
  --snapshot-id ref-model-2026-08 \
  --created-by operator \
  --dry-run
```

Operator 确认 endpoint 是可信参考源后执行真实捕获（复用 `llmtrace run` 的统一执行链）：

```bash
llmtrace reference capture \
  --protocol openai \
  --base-url https://api.example.com/v1 \
  --model reference-model \
  --api-key-env MY_API_KEY \
  --provider-id operator \
  --snapshot-id ref-model-2026-08 \
  --created-by operator \
  --yes
```

资格门禁（Gate 1–10）全部通过后生成 ReferenceSnapshot；任一失败则只保留运行工件、不生成快照。
从已验证快照构建 ReferenceSet（0 API 请求）：

```bash
llmtrace reference set-create \
  --reference-dir references \
  --set-id refset-v1 \
  --set-version 1.0.0 \
  --snapshot ref-model-2026-08 \
  --snapshot ref-model-2026-07
```

> `ReferenceSet` 当前不产生 calibrated 0–100 分数（v0.4-B Calibration 规划中）。

## 模拟服务器

项目包含一个本地模拟服务器，用于不消耗 API 的端到端验证：

```bash
python examples/mock_proxy_server.py --mode honest --port 8080
```

支持三种模式：
- `honest`：正常模型返回 v1，无效模型返回 404
- `fallback`：不论请求什么模型都成功返回 v1（无效模型也被接受），暴露静默回退行为
- `inconsistent`：正常模型在 v1/v2 之间轮换，无效模型返回 404；usage 字段时有时无

模拟服务器提供 `GET /debug/requests` 调试端点，返回实际接收到的审计请求日志（该端点本身不计入请求数），用于端到端核验 dry-run 计划次数与实际请求次数一致。

## 报告示例

每次审计生成同名 JSON 和 HTML 报告。JSON 报告包含完整脱敏证据，HTML 报告可本地打开，不依赖 CDN。

## 密钥安全

- 密钥仅允许来自环境变量或隐藏交互式输入
- 禁止命令行明文 `--api-key`
- 报告自动脱敏所有鉴权信息
- 不写入日志或异常堆栈

## 当前路线图

| 版本 | 内容 |
|------|------|
| v0.1 | 证据审计 MVP（已完成） |
| v0.2 | 轻量能力基准（已完成：lm-eval Adapter, GSM8K acceptance, Capability Scoring 基础） |
| v0.3-A | Item-Level Benchmark Foundation（已完成） |
| v0.3-B | Quick Suite 32 题 4 维度（已完成） |
| v0.3-C | Reference Model Snapshot（已完成） |
| v0.3-D | Behavior Drift Foundation（已完成） |
| v0.3-E | Unified Execution & Artifact Foundation（已完成：`llmtrace run`） |
| v0.4-A | Trusted Reference Run & Reference Set Foundation（已完成：`llmtrace reference capture / set-create`，无 0–100 输出） |
| v0.4-B | Reference Calibration & 0–100（规划中） |
| v0.5 | Fingerprint + Routing |
| v0.6 | Product Service / Web |

详细权威路线以 [`docs/roadmap.md`](./docs/roadmap.md) 为准。