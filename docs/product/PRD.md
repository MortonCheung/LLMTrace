# LLMTrace 产品需求文档（PRD）

> 产品名称：LLMTrace  
> 中文名称：模型寻迹  
> 文档版本：1.0  
> 当前代码版本：v0.3-E  
> 当前阶段：v0.4 Reference + Calibration IN PROGRESS；v0.4-A Trusted Reference Run & Reference Set Foundation 已完成，v0.4-B Reference Calibration & Claimed Model Gap 已完成（`llmtrace run --reference-set`）  
> 仓库：`MortonCheung/LLMTrace`  
> 当前开发基线：`main`，v0.1 证据审计 MVP、v0.2 基准评测基础设施、v0.3-A/B/C/D/E 全部已完成

---

## 1. 产品摘要

LLMTrace 是一个面向第三方 AI API、中转站和代理服务的黑盒能力审计工具。

用户输入中转站地址、API Key、声明模型和推理强度，点击“开始测试”后，系统自动完成：

1. 协议与连通性检查；
2. 标准化能力评测；
3. 0–100 分能力评分；
4. 与声明模型参考成绩对比；
5. 行为相似度分析；
6. 动态路由、混合模型和能力降级检测；
7. 生成普通用户可理解、专业用户可复核的报告。

### 一句话定位

> 将学术界、研究机构和开源社区严肃、复杂、门槛高的大模型评测能力，封装成普通用户可以直接使用的第三方 AI API 验货工具。

### 产品原则

LLMTrace 不重新发明已经成熟的学术基准。外部项目和研究负责题目、评分、统计、沙箱与学术可信度；LLMTrace 自研：

- 中转站协议适配；
- 外部评测框架适配；
- 测试计划编排；
- 请求、Token、时间与费用预算；
- 结果标准化；
- 参考模型档案；
- 声明差距计算；
- 行为相似与路由分析；
- 大众化解释；
- 统一报告和简易前端。

---

## 2. 当前开发基线

### 2.1 已完成：v0.1 → v0.3-E

当前仓库已完成五条纵向闭环：

```text
用户配置
→ Provider 发送请求
→ 采集 HTTP Evidence
→ Probe 分析
→ Risk Analysis
→ JSON / HTML Report
```

```text
用户配置
→ 统一执行计划（UnifiedAuditRunner：PRECHECK → PLAN → PROTOCOL → BENCHMARK → SCORING → SNAPSHOT → 比较 → REPORT/ARTIFACT COMMIT）
→ Quick Suite 32 题（llmtrace_quick_v1）
→ CapabilityProfile（UNCALIBRATED）
→ Behavior Drift / Reference Comparison
→ append-only Run Artifact
```

已完成能力（对应版本）：

- v0.1 证据审计 MVP：OpenAI/Anthropic-compatible 协议适配、探针、模型标识漂移检测、JSON/HTML 报告、CLI `audit` / `inspect` / `compare`
- v0.2 评测粘合层与能力评分基础：Source/Suite/Task/Run/Score 统一数据模型、Benchmark Adapter Protocol、`lm-evaluation-harness` 最小适配器、四维能力评分引擎（uncalibrated）
- v0.3-A Item-Level Benchmark Foundation：`BenchmarkItemResult` 逐题评分、`GradeResult` 状态模型、CodeExecutionBackend（Docker sandbox）
- v0.3-B Quick Suite：四维 32 题 `llmtrace_quick_v1`、不可变固定子集（SHA-256 排名）、`coverage_weight = 0.75`
- v0.3-C Reference Model Snapshot：`ReferenceSnapshot` / `ReferenceRepository` / `CapabilityComparator`，不可变、追加式参考画像
- v0.3-D Behavior Drift Foundation：`BehaviorRunSnapshot` / `BehaviorDriftEngine` / 四类 Detector，稳定身份逐项对齐
- v0.3-E Unified Execution & Artifact Foundation：`llmtrace run` 一键真实审计、`UnifiedAuditRunner`、`RunArtifactRepository`、`RequestBudget` / `EvidenceRecorder`、Base URL 与 secret 脱敏

### 2.2 当前缺口

尚未完成：

- 官方参考模型成绩库规模化与 LiteLLM ReferenceProvider 集成（v0.4 遗留，后续版本）；
- 行为相似度 / 指纹（v0.5）；
- 混合路由与动态降级（v0.5）；
- 动态网络来源更新；
- 后端任务服务、SQLite 历史记录、简易 Web 页面（v0.6）。

### 2.3 进度判断

```text
协议审计 MVP：100%
参考运行 / ReferenceSnapshot / ReferenceSet 基础设施：已落地（v0.4-A）
0–100 校准与声明模型差距：已落地（v0.4-B：piecewise anchor CalibrationPolicy v1 + Claimed Model Gap，fail closed）
能力验货闭环（含 Calibration）：核心已达成；剩余官方参考成绩库规模化
完整产品（后端 + 参考体系 + 相似度 + 简易前端）：进行中
```

---

## 3. 产品背景

第三方 AI API 和中转站存在典型问题：

- 用户只能看到服务商声明的模型名；
- 中转站可以改写 `model` 字段；
- 可能忽略模型参数或静默回退；
- 不同请求可能走不同上游；
- 高峰期可能切换到低成本模型；
- 厂商和排行榜测试条件不统一；
- 现有学术评测工具安装复杂、依赖重、面向研究人员；
- 普通用户难以理解 accuracy、pass@1、Elo、MMD 和置信区间。

LLMTrace 解决的是：

```text
填写接口
→ 选择测试档位
→ 查看预计费用
→ 点击开始
→ 得到清晰、可复核的验货报告
```

---

## 4. 目标用户

### 4.1 中转站普通用户

关心：

- 实际模型大概什么水平；
- 和宣传模型差多少；
- 有没有降级；
- 接口是否稳定。

### 4.2 API 开发者

关心：

- 不同服务商的能力、速度、成本和稳定性；
- 不同能力维度差异；
- 接口更新后的回归。

### 4.3 中转站运营者

关心：

- 上游稳定性；
- 路由质量；
- 高峰期降级；
- 可公开验证的质量报告。

### 4.4 AI 产品团队

关心：

- 候选 API 的业务适配；
- 长期趋势；
- 成本、速度和能力的综合比较。

---

## 5. 产品目标与非目标

### 5.1 必须实现

1. 输入中转站信息后自动测试；
2. 输出 0–100 综合能力分；
3. 输出分项能力分；
4. 输出与声明模型的差距；
5. 输出 Top-K 行为相似参考模型；
6. 输出路由稳定性和疑似混合路由；
7. 输出置信度与置信区间；
8. 输出请求数、Token、耗时和费用；
9. 所有结论可追溯到 Evidence；
10. 所有标准、题库、评分器和参考数据版本化；
11. 普通用户看到直白解释；
12. 专业用户可以查看原始指标。

### 5.2 明确不做

- 不声称绝对确认真实上游模型身份；
- 不自行发明完整学术基准；
- 不直接相信厂商自报成绩；
- 不以 LLM-as-Judge 作为客观能力主分；
- 不让网络更新直接覆盖正式标准；
- 首版不做 SaaS、账户、支付和订阅；
- 首版不一次接入所有评测框架。

---

## 6. 核心用户流程

### 6.1 配置

用户填写：

- Base URL；
- API Key；
- 声明模型；
- 协议；
- 鉴权方式；
- 推理强度；
- 温度；
- 最大输出 Token；
- 测试档位；
- 最大预算。

系统自动：

- 识别或验证协议；
- 验证地址和密钥；
- 获取模型列表；
- 检测流式；
- 检查模型参数；
- 估算费用。

### 6.2 测试档位

#### 快速测试

- 20–30 次请求；
- 5–10 分钟；
- 推理、数学、代码、指令遵循；
- 给出大致能力档位。

#### 标准测试

- 50–100 次请求；
- 完整分项；
- 声明模型差距；
- 初步行为相似度。

#### 深度测试

- 150 次以上请求；
- 重复采样；
- 跨时间检测；
- 行为相似；
- 混合路由与动态降级。

### 6.3 开始前确认

必须显示：

- 预计请求数；
- 输入/输出 Token；
- 预计耗时；
- 预计费用；
- 是否运行代码；
- 是否重复采样；
- Suite 版本。

### 6.4 执行流程

```text
协议预检
→ 生成 RunPlan
→ 运行 Benchmark Adapter
→ 收集原始响应和 Evidence
→ 客观评分
→ 统一标准化
→ 参考模型对照
→ 行为特征
→ 路由分析
→ 统一报告
```

---

## 7. 输出结果

普通报告示例：

```text
综合能力：78.6 / 100
能力档位：高端模型水平
声明模型参考：89.4
实际差距：-10.8
主要短板：推理、长文本
行为最相似：参考模型 A、参考模型 B
路由稳定性：一般
疑似混合路由：中等可能
```

专业报告包括：

- 原始 Benchmark 指标；
- 每题得分；
- 失败原因；
- Evidence；
- 外部项目版本；
- Suite、Scoring、Reference Set 版本；
- 统计方法；
- 置信区间；
- Token、费用和延迟。

---

## 8. 外部评测项目接入

### 8.1 lm-evaluation-harness

首要通用适配器。

用途：

- 知识；
- 推理；
- 数学；
- 指令遵循；
- 统一学术任务执行。

LLMTrace 只负责 Provider、任务选择、执行参数、结果转换和 Evidence 关联。

### 8.2 LiveBench

动态题库主要来源。

用途：

- 更新较新的推理、数学、代码、语言、指令和数据分析题；
- 降低老公开题污染；
- 保持客观自动评分。

首版只接入精选、客观、低成本任务。

### 8.3 EvalPlus

代码能力首要适配器。

用途：

- HumanEval+ / MBPP+ 精选题；
- pass@1；
- 隐藏测试通过率；
- 编译和运行错误分类。

### 8.4 Inspect AI

第二阶段接入，用于：

- 工具调用；
- Agent；
- 多轮任务；
- 复杂环境；
- 安全评测。

### 8.5 主要作为设计参考

- Promptfoo：Provider、测试矩阵、断言、回归和报告；
- OpenAI Evals：Eval 注册、私有任务、Completion Adapter；
- OpenCompass：大规模配置、调度、中文评测；
- SWE-bench：深度代码模式；
- FastChat / MT-Bench / Chatbot Arena：人类偏好，不进入客观能力主分。

---

## 9. 学术方法接入

### 9.1 LLMmap

用于主动指纹题和行为特征。

输出只能使用：

- “行为最相似”；
- “更接近”；
- “相似度证据”。

不能输出绝对身份。

### 9.2 Model Equality Testing

用于：

- 目标 API 和参考模型输出分布比较；
- 重复采样；
- 双样本统计检验；
- 判断是否与声明模型参考分布相符。

### 9.3 Artificial Analysis 方法

吸收：

- 多基准综合指数；
- 版本化权重；
- 置信区间；
- 能力、速度和成本分开报告。

v0.4-B 落地情况：版本化 CalibrationPolicy（id+version）与"指数分是相对比较工具"的
语义已实现；置信区间（重复运行统计）留待 v0.5+。方法论调研与采纳/拒绝记录见
`docs/architecture/open_source_influences.md`。

### 9.4 Hugging Face Leaderboard 原则

参考模型必须在相同题目、顺序、参数、推理强度、工具权限和评分器下运行。

---

## 10. 评分体系

### 10.1 最终维度与权重

| 维度 | 权重 |
|---|---:|
| 推理与专业知识 | 25% |
| 编程 | 20% |
| 数学与科学 | 15% |
| 指令遵循 | 15% |
| 数据分析 | 10% |
| 长文本 | 10% |
| 工具调用 | 5% |

v0.3 能力评分 MVP 只实现前四项。

### 10.2 评分优先级

1. 程序执行；
2. 精确答案；
3. 数值容差；
4. 集合匹配；
5. JSON Schema；
6. 约束满足率；
7. 规则评分；
8. 模型裁判。

### 10.3 原始指标

必须保留：

- accuracy；
- pass@1；
- unit test pass ratio；
- strict instruction accuracy；
- exact match；
- task completion rate。

### 10.4 100 分归一化

所有分数绑定：

- `suite_version`；
- `scoring_version`；
- `reference_set_version`。

建议语义：

```text
0 分：无有效回答或接近随机
50 分：参考模型组中位能力
90 分：旗舰参考组水平
100 分：当前 Suite 保留上限
```

不允许把某个模型永久写死为 90 分。

### 10.5 置信度

依据：

- 完成题数；
- 维度覆盖；
- 请求失败率；
- 重复采样波动；
- 参考模型覆盖；
- 题目难度分布。

输出：

```text
综合能力：82.4 / 100
95% 置信区间：79.8–84.7
置信度：中高
Suite：capability-2026.08
```

---

## 11. 参考模型体系

### 11.1 ReferenceProfile

记录：

- 供应商和模型；
- 模型快照；
- 推理强度；
- 温度；
- 最大 Token；
- 工具权限；
- Suite 与 Scoring 版本；
- 时间；
- 原始和分项成绩；
- 请求数、Token、成本；
- 重复次数。

### 11.2 参考数据分级

- **LLMTrace 实测**：可用于正式差距和相似度；
- **厂商官方报告**：只保存为 `vendor_claimed`；
- **第三方排行榜**：用于合理区间和辅助校准。

### 11.3 声明模型差距

只有模型版本、推理强度、温度、工具权限和 Suite 基本匹配时才计算。

示例：

```text
声明模型参考分：90.3
中转站实测分：78.8
综合差距：-11.5

推理：-15.2
代码：-4.8
数学：-12.1
指令遵循：-8.7
```

没有匹配参考配置时，只提供能力档位。

---

## 12. 行为相似度

行为向量包括：

- 逐题正确/错误；
- 分项结构；
- 格式遵循；
- 拒答模式；
- 答案长度；
- 自我修正；
- 代码和数学错误类型；
- 约束违反方式；
- Token；
- 延迟；
- 主动指纹题；
- 流式行为。

输出：

```text
模型 A / High：64%
模型 B / Standard：23%
模型 C / Max：8%
无法归类：5%
```

固定提示：

> 该结果表示行为相似，不代表已确认真实上游模型身份。

---

## 13. 混合路由与动态降级

判断依据：

- 同题重复；
- 不同能力类别；
- 不同时间段；
- 输出聚类；
- 能力突变；
- 延迟和 Token 分布；
- 模型标识变化；
- Model Equality Testing；
- 跨报告漂移。

输出：

```text
路由稳定性：一般
疑似混合路由：中等可能
数学任务更接近参考模型 A
代码任务更接近参考模型 B
样本不足以确认精确混合比例
```

---

## 14. 动态网络来源系统

### 14.1 来源等级

| 等级 | 来源 | 用途 |
|---|---|---|
| A | 同行评审、客观评分、可复现 | 可进入候选 Suite |
| B | 成熟开源框架、正式排行榜 | 可进入候选或参考库 |
| C | 厂商技术报告 | 只保存声明成绩 |
| D | 媒体、社区 | 只用于发现来源 |

### 14.2 Source Registry

```yaml
source_id: livebench
source_type: benchmark
repository: LiveBench/LiveBench
revision: commit-sha
license: Apache-2.0
trust_tier: A
update_policy: monthly
last_checked_at: timestamp
checksum: sha256
```

### 14.3 更新流程

```text
在线发现
→ 下载候选区
→ 许可证检查
→ 固定 commit
→ 格式校验
→ 去重与质量检查
→ 参考模型试跑
→ 新旧标准比较
→ 人工批准
→ 发布新 SuiteVersion
```

正式 Suite 发布后不可变。

---

## 15. 后端架构

保留现有：

```text
providers/
probes/
analysis/
reporting/
security/
models/
utilities/
```

新增：

```text
src/llmtrace/
├── benchmarks/
│   ├── registry.py
│   ├── models.py
│   ├── planner.py
│   └── suites/
├── adapters/
│   ├── base.py
│   ├── lm_eval.py
│   ├── livebench.py
│   ├── evalplus.py
│   └── inspect.py
├── grading/
│   ├── exact.py
│   ├── numeric.py
│   ├── schema.py
│   ├── constraints.py
│   └── code.py
├── scoring/
│   ├── dimensions.py
│   ├── normalization.py
│   ├── confidence.py
│   └── capability.py
├── references/
│   ├── registry.py
│   ├── profiles.py
│   ├── calibration.py
│   └── comparison.py
├── fingerprint/
│   ├── features.py
│   ├── similarity.py
│   ├── equality.py
│   └── routing.py
├── sources/
│   ├── registry.py
│   ├── updater.py
│   ├── validation.py
│   └── licenses.py
├── execution/
│   ├── service.py
│   ├── runner.py
│   ├── budget.py
│   ├── progress.py
│   └── cancellation.py
└── storage/
    ├── database.py
    ├── runs.py
    └── artifacts.py
```

架构语义：

```text
外部项目 = 测量仪器
Adapter = 插头
Execution Service = 调度台
Scoring = 统一刻度
ReferenceProfile = 标准样本
Reporting = 用户说明书
```

---

## 16. 核心数据模型

必须定义：

- `EndpointConfig`
- `BenchmarkSource`
- `BenchmarkSuite`
- `SuiteVersion`
- `TaskSpec`
- `RunPlan`
- `TaskAttempt`
- `GradeResult`
- `CapabilityVector`
- `CapabilityScore`
- `ReferenceProfile`
- `SimilarityResult`
- `RoutingAnalysis`
- `AssessmentReport`

每次测试保存：

- 题库版本；
- 外部项目 commit；
- Adapter 与评分器版本；
- 模型参数；
- 每题 Evidence；
- 原始分与标准化分；
- 费用、Token、时间；
- 失败原因。

---

## 17. 服务层与前端接口

后端核心不得依赖 CLI。

统一服务：

```python
create_run(config)
estimate_run(run_id)
start_run(run_id)
get_progress(run_id)
cancel_run(run_id)
get_result(run_id)
```

FastAPI：

```text
POST /api/runs
POST /api/runs/{id}/estimate
POST /api/runs/{id}/start
GET  /api/runs/{id}
GET  /api/runs/{id}/events
POST /api/runs/{id}/cancel

GET /api/suites
GET /api/sources
GET /api/reference-models
```

进度使用 SSE。

首版存储：

- SQLite；
- JSON；
- HTML；
- 本地文件。

不引入 Redis、消息队列和云数据库。

---

## 18. 简易前端范围

### 页面一：开始测试

- Base URL；
- API Key；
- 声明模型；
- 协议；
- 推理强度；
- 测试档位；
- 预算；
- 成本估算；
- 开始按钮。

### 页面二：执行进度

- 当前阶段；
- 已完成任务；
- 请求数；
- Token；
- 剩余时间；
- 取消；
- 日志摘要。

### 页面三：结果报告

- 综合分；
- 分项能力；
- 声明差距；
- 相似模型；
- 路由稳定性；
- 协议风险；
- 成本和速度；
- 专业数据；
- JSON / HTML 导出。

首版不做账户、支付和复杂视觉。

---

## 19. 安全要求

1. API Key 默认只在内存；
2. 不在命令行明文接收；
3. 不写日志和报告；
4. Base URL 防 SSRF；
5. 禁止访问本机敏感地址；
6. 代码题隔离执行；
7. 请求、Token、费用和时间硬上限；
8. 用户可取消；
9. 外部题库代码未经审核不得执行；
10. 网络更新只进入候选区；
11. 原始响应默认不上传；
12. 默认本地运行；
13. HTML 防 XSS；
14. 错误信息脱敏；
15. 用户凭据和参考模型凭据隔离。

---

## 20. 测试要求

### 单元测试

覆盖 Adapter、Registry、Planner、Budget、Grader、Normalization、Confidence、Reference Comparison 和报告序列化。

### 集成测试

使用 Mock Provider 验证：

```text
RunPlan
→ Adapter
→ Evidence
→ Grade
→ Score
→ Report
```

### Adapter 契约测试

每个 Adapter 必须：

- 可加载；
- 可列任务；
- 可生成计划；
- 可执行；
- 返回统一结果；
- 关联 Evidence；
- 报告版本；
- 处理失败；
- 支持取消；
- 估算成本。

### Golden Suite

固定一个不调用真实 API 的小型套件，CI 每次运行，保证结果和报告结构稳定。

---

## 21. 开发里程碑

### Phase 0：质量基线

- GitHub Actions；
- PRD 入库；
- 更新 roadmap；
- Python 3.11 / 3.12 / 3.13；
- Ruff、Format、Mypy、Pytest；
- 覆盖率门禁；
- Mock Server 集成测试。

### Phase 1：评测粘合层

- Source、Suite、Task、Run、Score 数据模型；
- Adapter Protocol；
- RunPlan；
- Budget Estimator；
- 统一结果模型；
- 首个 `lm-evaluation-harness` 适配器。

完成标准：一个外部任务通过 Mock Provider 跑通并进入 Evidence 与报告。

### Phase 2：能力评分 MVP

接入：

- lm-eval 精选任务；
- LiveBench 精选客观题；
- EvalPlus 精选代码题。

实现：

- 快速模式；
- 4 个能力维度；
- 20–40 个任务；
- 0–100 分；
- 置信度；
- 成本估算；
- JSON / HTML 报告。

### Phase 3：参考模型对照

- ReferenceProfile；
- 官方 API 实测；
- 声明差距；
- 分项差距；
- 版本化参考组。

### Phase 4：行为相似与路由

- 主动指纹题；
- 行为向量；
- Top-K 相似；
- Model Equality Testing；
- 混合路由；
- 跨时间降级。

### Phase 5：简易前端

- 输入接口；
- 成本确认；
- 开始测试；
- 实时进度；
- 结果报告；
- 历史记录。

---

## 22. v0.3 能力评分 MVP 验收标准

1. 支持配置任意 OpenAI-compatible 或 Anthropic-compatible 接口；
2. 可生成测试计划；
3. 可估算请求数、Token、时间和费用；
4. 可执行至少一个外部 Adapter；
5. 可运行 20–40 个客观任务；
6. 输出推理、数学、代码、指令遵循分数；
7. 输出 0–100 综合分；
8. 输出置信度；
9. 每个任务有 Evidence；
10. 报告记录 Suite、Scoring、Adapter 和外部项目版本；
11. 单个请求失败不破坏全局；
12. 用户可取消；
13. Mock Provider 集成测试通过；
14. CI 通过；
15. 不混入前端、指纹和混合路由半成品。

---

## 23. 主要风险

### 数据污染

使用动态题、LiveBench、私有题和多来源组合。

### 参考配置不一致

ReferenceProfile 严格匹配；不匹配时不输出精确差距。

### 单一总分误导

总分与分项同时展示，绑定 Suite 版本和置信区间。

### 中转站针对评测优化

动态变体、私有题、随机化和跨时间运行。

### 许可证

Source Registry、License Validator 和人工审核。

### 代码执行安全

Docker 或等价隔离、无网络、只读文件系统、CPU/内存/时间限制。

### API 成本

执行前估算、硬预算、可取消、三档测试。

### 相似度误解

固定免责声明，不输出绝对身份。

---

## 24. 未决问题

1. 第一版 lm-eval 精选哪些任务；
2. 快速模式成本上限；
3. 100 分归一化采用锚点、分位数还是混合方法；
4. 第一批参考模型范围；
5. 是否允许用户提供参考 API；
6. 官方参考运行成本来源；
7. 动态题和私有题如何分发；
8. EvalPlus 使用 Docker 还是独立进程；
9. 中文能力是否独立维度；
10. 前端采用嵌入式本地 Web 还是独立 React；
11. 是否公开参考模型原始回答；
12. 网络更新是否自动创建待审 PR。

---

## 25. 当前开发决策

本阶段不做：

- 前端；
- 行为指纹；
- 混合路由；
- 大规模参考模型运行；
- 多框架同时接入。

本阶段只做：

```text
CI
→ PRD 与 roadmap
→ Benchmark Adapter Protocol
→ Source / Suite / Task / Run / Score 模型
→ lm-evaluation-harness 最小适配器
→ Mock Provider 单任务闭环
→ Evidence 与 Report 集成
```

本阶段完成后，项目才正式从“协议审计工具”进入“可扩展的大模型能力评测平台”。

---

## 26. 产品结论

LLMTrace 的竞争力不来自重新发明 Benchmark，而来自：

- 把成熟评测接入中转站；
- 统一编排不同项目；
- 将结果放进同一套可解释坐标；
- 严格区分官方声明、第三方成绩和 LLMTrace 实测；
- 将专业结论转成普通用户看得懂的语言；
- 保留完整 Evidence；
- 识别中转站特有的静默降级、混合路由和跨时间漂移。

最终定义：

> **LLMTrace 是面向普通用户的第三方 AI API 黑盒验货工具，以可信开源评测和学术研究为测量基础，以自研粘合层完成协议接入、任务编排、结果标准化、参考对照、行为分析和大众化报告。**
