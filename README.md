# LLMTrace - 模型寻迹

面向第三方 AI API、中转站和代理服务的黑盒模型审计工具。

## Quick Start（CLI）

```bash
git clone https://github.com/MortonCheung/LLMTrace.git
cd LLMTrace
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

export LLMTRACE_API_KEY="your-key"
llmtrace run \
  --base-url https://api.example.com/v1 \
  --model claimed-model \
  --yes
```

最小参数集：只需 Base URL、声明模型和一个 API Key 环境变量（默认 `openai` 协议与
`LLMTRACE_API_KEY`，可用 `-p`/`-k` 覆盖）。`llmtrace run` 会展示执行计划、实时进度
并给出带回溯证据的审计报告。想先看计划不发送请求：加 `--dry-run`；
本机环境异常先执行 `llmtrace doctor`。

### Web 界面（可选）

```bash
llmtrace web
```

打开 http://127.0.0.1:8765 开始使用：New Audit 表单 → 实时进度（SSE）→ 结果报告 → 历史记录。
默认只绑定 127.0.0.1（页面会接收 API Key）。

不想消耗真实 API、只想体验完整流程时加 `--demo`（进程内 Mock 端点 + Quick Suite 32 题，
页面带 DEMO 标识）：

```bash
llmtrace web --demo
```

要求 Python 3.11+。

## 当前版本能做什么

当前已完成 v0.1 证据审计 MVP、v0.2 基准评测基础设施、v0.3 能力评测收口
（item-level 结果、Quick Suite 32 题、Reference Model Snapshot、Behavior Drift Foundation）、
v0.3-E 一键统一审计执行链（`llmtrace run`）、v0.4 受信任参考体系与正式 0–100 校准
（v0.4-A ReferenceSet + v0.4-B Reference Calibration & Claimed Model Gap）、v0.5 Usable MVP
（本地 Web 应用 + 持久 Run 索引 + Demo 模式），以及 v0.6 Identity Evidence Foundation
（实验性行为身份证据 + routing v2 证据分级；CLI-first）。

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

> 注意：`ReferenceSet` 本身**不产生 calibrated 0–100 分数**——参考快照/参考组是可信、
> 不可变、可审计的历史能力事实。正式 0–100 能力分由 v0.4-B Calibration 在运行时
> 通过 `--reference-set` 对候选测量推导（见下节），两者是不可混淆的概念。

### v0.4-B Reference Calibration & 0–100（实验性，新增）

- **正式 0–100 Capability Score**：`llmtrace run --reference-set` 在统一审计后对
  raw 能力画像做 Reference Anchored Monotonic Piecewise Calibration——
  `0 = 随机基线`、`50 = 参考组中位数`、`90 = P90（flagship）`、`100 = 套件上限`
  的版本化分段线性锚点映射（`llmtrace-reference-calibration-v1`）
- **Claimed Model Gap**：被测 endpoint 与声明模型的兼容可信参考配置之间的
  总分差距与分维度差距；严格 model_id 匹配（无模糊匹配、歧义即拒绝）；
  **是能力对比，不是模型身份证明**
- **Fail-closed**：参考身份 < 5、离散度不足、校准饱和、候选测量不完整、
  scoring policy 不一致、ReferenceSet 信任链验证失败——任一触发即保持
  `UNCALIBRATED` 并输出 warning，绝不产生无依据分数；信任链失败在 preflight
  即拒绝（0 次 HTTP 请求）
- **Provenance 贯穿**：manifest / JSON / HTML 报告记录 calibration policy
  id+version、reference set id+version+content SHA；raw 分数全部保留
- **语义边界**：0–100 分是相对某个已定义校准宇宙（ReferenceSet + Policy）的
  相对坐标，不是绝对能力；Reference Comparison（原始对比）≠ Calibration（正式映射）；
  分数不可跨参考组比较

### v0.5 Usable MVP（新增）

- **本地 Web 应用**：`llmtrace web`（默认 `127.0.0.1:8765`，页面会接收 API Key）——
  New Audit 表单 → 实时进度（SSE）→ 结果报告 → 历史记录，Jinja + Vanilla JS，无外部前端依赖
- **RunService**：进程内单例管理 run 生命周期（create / estimate / start / cancel / progress /
  result / history），后台任务 + 事件环形缓冲 + cooperative cancellation
- **持久 Run 索引**：SQLite 记录每次 run 的状态与终态摘要（不含任何 secret）
- **Demo 模式**：`llmtrace web --demo` 内置进程内 Mock 模型端点与 Quick Suite 32 题，
  零 API Key、零外部请求体验完整流程（页面带 DEMO 标识）
- **本机直连修复**：目标为 localhost/127.x 时 Provider 关闭环境代理 trust_env，
  避免 HTTP_PROXY 劫持回环请求；远端端点仍保留 httpx 默认代理行为

### v0.6 Model Verification · Experimental（新增）

> **Experimental。** v0.6 引入的是一层**行为身份证据（behavioral identity evidence）**，
> 用来回答"本次观测到的行为分布，是否与**受信任指纹参考集中被声称身份**的参考分布一致"。
> 它**不是**身份证明，也**不是**密码学证明。

```bash
export LLMTRACE_API_KEY="your-key"
llmtrace run \
  --base-url https://api.example.com/v1 \
  --model claimed-model \
  --verify-model \
  --fingerprint-profile standard
```

`--verify-model` 在常规能力审计之外追加指纹采集阶段，并在报告中输出：

- **Claim Consistency**（`BEHAVIOR CONSISTENT WITH CLAIM` / `BEHAVIOR INCONSISTENT WITH CLAIM`）
- **Fingerprint Confidence**（`HIGH` / `MEDIUM` / `LOW` / `UNAVAILABLE`）
- **Top Behavioral Matches**（Top-K：距离最接近的受信任参考身份，距离越小 = 行为分布越接近）

`--fingerprint-profile` 只是**成本档位**（`quick` / `standard` / `research` = 4 / 8 / 16 probes ×
重复轮数），**不代表准确率**。

#### 受信任指纹参考集与自动发现

身份证据需要一个**受信任指纹参考集**（`FingerprintReferenceSet`，由 Operator 在受信任端点上采集）。
建立信任链的三步（均为独立 CLI，0 次能力评测请求）：

```bash
# 1) 在受信任端点上采集指纹快照（Operator 断言；重复多轮）
llmtrace fingerprint capture \
  --base-url https://trusted.example.com/v1 \
  --model trusted-model \
  --api-key-env MY_API_KEY \
  --role official-baseline \
  --profile standard \
  --snapshot-id trusted-model-2026-09-01 --yes

# 2) 由快照构建参考集（0 API 请求）
llmtrace fingerprint set-create \
  --set-id my-identities --set-version 1.0.0 \
  --snapshot trusted-model-2026-09-01 --snapshot trusted-model-2026-09-02

# 3) 在参考集上做 held-out 验证，生成 decision policy（0 API 请求）
llmtrace fingerprint validate \
  --set-id my-identities --set-version 1.0.0 \
  --policy-id my-policy --policy-version 1.0.0
```

参考集**自动发现**：`llmtrace run --verify-model` 会自动扫描指纹仓库目录
（默认 `~/.llmtrace/fingerprints/sets/`，可用 `--data-dir` / 环境变量改根目录），
只保留**能完整通过校验**（套件身份 + 重复轮数 + 成员快照完整性 + 自哈希）的集合。
优先级为 `显式 --fingerprint-set` > **唯一**兼容自动发现 > 无；存在多个兼容集合时
**绝不随机挑选**，非交互环境 fail closed 要求显式指定。

**没有兼容的 `FingerprintReferenceSet` 时：Model Verification = `Unavailable`，
但正常 Capability Audit 继续执行。** 指纹阶段被跳过的原因是可审计的（缺参考集、
参考集未通过校验、policy 未通过 held-out 验证等），不会把一次 `run` 变成失败。

#### 判定规则只能来自 LLMTrace 自己的 held-out 验证

只有 **`validated == True`** 的 decision policy 才允许输出 `BEHAVIOR CONSISTENT WITH CLAIM` /
`BEHAVIOR INCONSISTENT WITH CLAIM`；否则报告只会给出 Top-K 排序
（`RANKED_ONLY`，有排序、无判定）。threshold 由 `llmtrace fingerprint validate`
在参考集上以 leave-one-capture-out 选出，并受 `false_accept_rate <= --max-far` 约束；
**不通过验证的数据集不产生 threshold、也不发布准确率数字。**

#### 三层语义边界（必须同时成立）

| 表述 | 含义 |
|---|---|
| **Capability Score ≠ Identity Verification** | 能力分（含 0–100 校准分）是能力坐标，不是身份结论 |
| **Behavior Similarity ≠ Fingerprint Match** | Top-K 距离接近是排序证据；只有 validated policy + 被声称身份的参考才产生 claim consistency |
| **Fingerprint Match ≠ Cryptographic Proof of upstream identity** | 行为证据可被规避、可因服务端采样/版本变化而漂移，永远不是身份证明 |

推荐用语：**behavioral identity evidence**、**claim consistency**、**trusted fingerprint reference**、
**experimental model verification**。

#### Routing v2（实验性）

`--verify-model` 的报告中另含 routing 信号分级：**Strong**（多个稳定 `response_model` 标识且
无单一主导 / 已验证参考匹配在时间窗之间切换 / 候选时间散度超过已验证参考基线）与
**Supporting**（延迟离散度 / token 离散度 / provider 失败率）。Supporting 信号**单独**不足以
得出"检测到混合路由"。`Suspicious` 是**审计结论**，**不会**把 run 变成 failure
（run status 仍为 `COMPLETED`）。LLMTrace 只支持"观察到不一致"与"可能存在混合路由"两层，
**不声明**已证明路由混合比例。

详见：[`docs/methodology/fingerprinting.md`](./docs/methodology/fingerprinting.md)、
[`docs/methodology/routing.md`](./docs/methodology/routing.md)。

## 语义边界：LLMTrace 能说什么、不能说什么

- ✅ 能说：「在相同测试条件下，本次运行与历史运行观察到显著行为漂移。」
- ❌ 不能说：「模型被确定偷换成了 XXX。」
- ✅ 能说：「输出行为发生变化。」
- ❌ 不能说：「输出文本不同，因此底层模型不同。」
- ✅ 能说：「能力结果显著下降。」
- ❌ 但如果主要原因是 Provider Failure，不能说：「模型能力显著下降。」
- ✅ 能说：「相对参考组 X（v1），本次实测能力分 72.5，比声明模型的可信参考低 8 分。」
- ❌ 不能说：「实测分数低于声明模型参考，所以服务商偷换了模型。」（能力对比 ≠ 身份证明）
- ✅ 能说：「本次行为分布与受信任指纹参考集中**被声称身份**的参考分布一致 / 不一致。」（claim consistency，实验性）
- ❌ 不能说：「已经证明上游就是该模型。」或「已经证明服务商跑了假模型。」
- ✅ 能说：「观察到 routing inconsistency：本次样本中出现多个无单一主导的 `response_model` 标识，混合路由是可能的。」
- ❌ 不能说：「已证明该中转站按 70% A / 30% B 混合路由。」（比例不可证明）

## 当前版本不能证明什么

- 不能识别中转站真实上游模型：v0.6 提供的是**实验性行为身份证据**（claim consistency），
  它能说"行为分布是否与受信任参考一致"，不能说"真实上游模型是什么"
- Fingerprint 行为匹配**不是密码学证明**，也不构成对上游身份的证明
- 没有兼容的 `FingerprintReferenceSet` 时，Model Verification 为 `Unavailable`（能力审计照常继续）
- 不能证明服务商是否使用了声明模型
- LOW 风险等级只表示"本次有限测试未发现明显异常，不代表已证明真实上游模型身份"
- 单次延迟不能直接判断模型身份
- 输出文本不同不代表底层模型不同；行为相似度不是身份结论
- Provider 失败（超时/限流）不等价于模型能力下降

## 使用示例

### 一键统一审计（推荐）

```bash
export LLMTRACE_API_KEY="your-key"
llmtrace run \
  --base-url https://api.example.com/v1 \
  --model claimed-model \
  --yes
```

一次 `run` 大约包含：协议探针（若干）+ Quick Suite 32 题 benchmark。实际请求数以
`--dry-run` 为准；运行前会显示预计请求数、最大输出 token ceiling 与预计费用（未知），
确认后才执行。`--yes` 跳过确认；`--compare-latest`（默认）自动与最新兼容历史运行做 Behavior Drift。
提供 `--reference-set`（ReferenceSet JSON 路径）时，额外输出正式 0–100 校准分与
声明模型差距（preflight 即重验参考组信任链，失败则 0 次 HTTP 请求直接拒绝）：

```bash
llmtrace run \
  --base-url https://api.example.com/v1 \
  --model claimed-model \
  --reference-set references/sets/refset-v1_0.1.0.json
```

```bash
# 只显示执行计划，不发送任何请求（0 HTTP、0 工件、不要求 API key 存在）
llmtrace run --base-url https://api.example.com/v1 --model demo --dry-run
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

> `ReferenceSet` 本身不产生 calibrated 0–100 分数；正式校准分在 `llmtrace run --reference-set`
> 运行时对候选测量推导（见上文 v0.4-B 一节）。

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
| v0.4-B | Reference Calibration & Claimed Model Gap（已完成：`llmtrace run --reference-set` 正式 0–100 校准 + 声明模型差距） |
| v0.5 | Usable MVP（已完成：本地 Web + RunService + SQLite Run 索引 + `--demo` Mock 模式） |
| v0.6 | Identity Evidence Foundation（已完成：categorical fingerprint probe suite + 重复采样 + JSD 分布匹配 + `FingerprintReferenceSnapshot` / `FingerprintReferenceSet` + held-out 验证 + `FingerprintDecisionPolicy` + claimed-model consistency + routing v2 + CLI 集成；实验性、CLI-first） |
| v0.7 | Relay Integrity / Protocol Conformance（未开始） |
| v0.8 | Claims Verification / Billing Integrity（未开始） |
| v0.9 | Evaluation Reliability / Standard-Deep（未开始） |

Deferred（v0.6 明确不做）：Web / Service 路径的 fingerprint 接线
（`RunService.start()` 的 `verify_model` / `fingerprint_profile` / `fingerprint_set_path`）。
**RunService fingerprint option wiring is deferred to the future Web/Service milestone.
The CLI execution path is the supported v0.6 identity-verification path.**

未来方向（先产品化，再精修）：

```text
MVP Hardening
Reference Data Expansion
Advanced Fingerprinting
Routing Statistics
Benchmark Expansion
Web Polish
```

研究性方向（仅登记，不在本轮实现）：Rank-Based Uniformity Test、LLMPrint 式更强统计验证、
natural-query 反规避探测、open-set model verification、mixture/routing 统计估计。

详细权威路线以 [`docs/roadmap.md`](./docs/roadmap.md) 为准。