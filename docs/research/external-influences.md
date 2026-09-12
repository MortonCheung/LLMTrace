# External Influences & License Ledger — v0.6 Model Identity Evidence / Fingerprint Foundation

## 目的

v0.6 引入「Model Identity Evidence / Fingerprint Foundation」——一套与能力评测**完全分离**的黑盒模型身份证据层，
用于观测第三方 API / 中转站在行为层面是否与声称的模型身份一致。

本台账记录：外部项目/论文对本设计的影响、**可采纳点**、**明确不采纳点**、**许可边界**。
唯一目的：确保 LLMTrace **不复制任何他人源码**，也不因"借鉴"而引入不兼容的许可证义务。

## 使用规则

1. **不复制源码**：本台账覆盖的**全部**项目仅作**设计思想**参考；LLMTrace 一律自行实现。
2. **默认自己实现**：不安装外部项目、不引入其 runtime、不照搬其数据模型。
3. **保留 attribution**：凡采纳的设计思想，在本文件与相关设计文档中保留对来源项目的点名致谢。
4. **技术梯级（Rule 5）**：实现优先复用现有 Provider（`openai_compatible` / `anthropic_compatible`）、
   Python stdlib（`urllib`、`sqlite3`、`statistics`、`json`）、现有依赖（httpx / typer / fastapi / sqlite）、
   透明统计方法（如 JSD）与 held-out 验证；**不引入** numpy / scipy / sklearn / embedding 数据库 / 新 HTTP client。
5. **许可证核验原则**：本台账的许可证结论均通过实际检索仓库 `LICENSE` / 仓库页核验并附来源 URL；
   无法核验的一律标注 `License: Unverified — requires manual confirmation`，**不猜测**。

## Permissive reuse candidates

以下项目的许可证为宽松许可（MIT / Apache-2.0 / BSD 类），未来若确需**复制源码**，仍是首选核查对象；
但本轮**默认仍只借鉴思想、自行实现**。注意：宽松许可 ≠ 无义务，复制时须保留版权与许可声明。

| Project | License |
|---|---|
| [zing](https://github.com/cenbonew/zing) | Apache-2.0 |
| [pasquini-dario/LLMmap](https://github.com/pasquini-dario/LLMmap) | MIT |
| [dreamor/llm-fingerprint](https://github.com/dreamor/llm-fingerprint) | Tool: MIT / Research data: CC-BY |
| [S1M0N38/llm-fingerprint](https://github.com/S1M0N38/llm-fingerprint) | MIT |
| [cisco-ai-defense/model-provenance-kit](https://github.com/cisco-ai-defense/model-provenance-kit) | Apache-2.0 |
| [sunblaze-ucb/llm-api-audit](https://github.com/sunblaze-ucb/llm-api-audit) | MIT |
| [xyzhu123/RUT](https://github.com/xyzhu123/RUT) | MIT |
| [felipemaiapolo/tinyBenchmarks](https://github.com/felipemaiapolo/tinyBenchmarks) | MIT |
| [dill-lab/micro-benchmarking-reliability](https://github.com/dill-lab/micro-benchmarking-reliability) | MIT (code we wrote) |
| [lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM) | Apache-2.0 |
| [MilkThink-Lab/RouterEval](https://github.com/MilkThink-Lab/RouterEval) | MIT |
| [UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai) | MIT |
| [modelscope/evalscope](https://github.com/modelscope/evalscope) | Apache-2.0 |
| [open-compass/opencompass](https://github.com/open-compass/opencompass) | Apache-2.0 |
| [LiveBench/LiveBench](https://github.com/LiveBench/LiveBench) | Apache-2.0 |
| [openai/evals](https://github.com/openai/evals) | MIT |
| [promptfoo/promptfoo](https://github.com/promptfoo/promptfoo) | MIT |
| [evidentlyai/evidently](https://github.com/evidentlyai/evidently) | Apache-2.0 |
| [langfuse/langfuse](https://github.com/langfuse/langfuse) | MIT Expat（`ee/` 目录除外） |
| [vllm-project/vllm](https://github.com/vllm-project/vllm)（vLLM Bench） | Apache-2.0 |
| [ray-project/llmperf](https://github.com/ray-project/llmperf) | Apache-2.0 |
| [metareason-ai/metareason-core](https://github.com/metareason-ai/metareason-core) | MIT |

## Copy-prohibited（红线）

**以下项目的源码严禁复制、严禁粘贴进 LLMTrace、严禁以任何形式引入其代码。**
只能阅读其**公开文档 / 论文**获取方法论层面的启发，然后 clean-room 自行实现。

| Project | License | 红线 |
|---|---|---|
| [toby-bridges/api-relay-audit](https://github.com/toby-bridges/api-relay-audit) | **AGPL-3.0-only** | 强 copyleft。复制/改编任何片段都可能触发整个 LLMTrace 的 AGPL 传染义务。**严禁复制源码。** |
| [inferock/inferock-bench](https://github.com/inferock/inferock-bench) | **FSL-1.1-ALv2**（`apps/inferock-bench`，source-available，后续转 Apache-2.0） | 非 OSI 开源、source-available。**主仓严禁复制源码。** |

补充红线：
- 即使存在第三方对 API Relay Audit 的 MIT 分叉，引用时**必须指向 `toby-bridges` 主仓**，不得用分叉规避 AGPL 义务。
- Inferock 仓库为分目录授权：`packages/measure` 为 Apache-2.0、`spec` 为 CC-BY-4.0。
  但按保守原则，**本台账将整个 Inferock 主仓视为 copy-prohibited**，不区分目录。

## Research only（仅方法论参考，无代码复用）

以下来源本轮仅作**方法论/统计思路**参考，不采纳其实现、不引入其代码。
纯论文若无代码许可证，标注 `N/A (research paper; check linked repo before any code reuse)`。

| Source | 性质 |
|---|---|
| [Rank-Based Uniformity Test (arXiv:2506.06975)](https://arxiv.org/abs/2506.06975) | 论文（ICLR 2026）；配套代码 RUT 为 MIT |
| [LLMmap (arXiv:2407.15847)](https://arxiv.org/abs/2407.15847) | 论文（USENIX Security 2025）；配套代码 MIT |
| [LLMPrint (arXiv:2509.25448)](https://arxiv.org/abs/2509.25448) | 论文（ACL 2026 Main）；配套仓库未见 LICENSE |
| [tinyBenchmarks (arXiv:2402.14992)](https://arxiv.org/abs/2402.14992) | 论文；配套代码 MIT |
| [MINCE (arXiv:2606.22826)](https://arxiv.org/abs/2606.22826) | 论文；未见公开代码仓库 |
| [Micro-Benchmark Reliability (arXiv:2510.08730)](https://arxiv.org/abs/2510.08730) | 论文（ICLR 2026）；配套代码 MIT |
| [LLMRouterBench (arXiv:2601.07206)](https://arxiv.org/abs/2601.07206) | 论文（CC BY 4.0）；仓库未见 LICENSE |
| [Your Agent Is Mine (arXiv:2604.08407)](https://arxiv.org/abs/2604.08407) | 论文（CC BY-NC-ND 4.0，ACM CCS 2026）；无独立公开代码仓库 |

---

# 条目

## LLM Provider Eval

Purpose:
疑为针对 LLM Provider 的评测类项目；未能定位到唯一对应的公开仓库或论文。

Adopt:
- 无（未核验到明确来源，不据未证实的描述做设计决定）。

Do not adopt:
- 无（同上）。

License:
Unverified — requires manual confirmation

Source:
未定位到可核验 URL；多次差异化检索未命中对应仓库。

Implementation:
clean-room design; no source copied.

## zing

Purpose:
LLM 指纹/身份识别方向的项目（通过探测响应分布对后端模型做归属判断）。

Adopt:
- 以"多次重复采样得到输出分布，再比较分布而非单次文本"的指纹思路。
- 结果表达为置信/不确定而非二元判定（与 LLMTrace fail-closed 语义一致）。

Do not adopt:
- 其运行时与数据模型（不引入外部 runtime / 存储）。
- 不采纳任何需要 embedding 模型或向量库的相似度路径。

License:
Apache-2.0

Source:
https://github.com/cenbonew/zing

Implementation:
clean-room design; no source copied.

## LLMmap

Purpose:
通过黑盒探测输出分布对 LLM 做"指纹化"识别，类比浏览器指纹；论文见 USENIX Security 2025。

Adopt:
- "用一组固定探测 prompt 的输出分布构成指纹向量"的总体框架。
- 指纹 = 可复现的探测集合 × 稳定统计量，而非依赖模型自报身份。
- 明确区分"未知模型"与"已收录模型"的开放集识别语义。

Do not adopt:
- 其聚类/降维依赖（需要 numpy/scipy/sklearn 与向量库）——LLMTrace 只用 Python stdlib + 透明统计。
- 大规模指纹库的采集与训练流程（超出 local-first 单机范围）。
- 不采纳"单次探测即可下结论"的任何捷径。

License:
MIT（代码仓库）；论文 arXiv:2407.15847 为 arXiv 非独家许可。

Source:
https://github.com/pasquini-dario/LLMmap ；https://arxiv.org/abs/2407.15847

Implementation:
clean-room design; no source copied.

## LLMPrint

Purpose:
通过"提示注入"手段对 LLM 进行指纹识别（Fingerprinting LLMs via Prompt Injection），ACL 2026 Main。

Adopt:
- "模型身份可被特定探测输入放大差异化"这一假设，用于设计 LLMTrace 的探测集。
- 探测结果需区分"有信息量"与"无信息量"，避免把噪声当信号。

Do not adopt:
- 提示注入类攻击性探测（LLMTrace 是审计工具，不做攻击、不诱导越权行为）。
- 其模型清单与专用建模流程。

License:
Unverified — requires manual confirmation（论文 arXiv:2509.25448；配套仓库 https://github.com/hifi-hyp/ACL-LLMPrint 目录中未见 LICENSE 文件）。

Source:
https://arxiv.org/abs/2509.25448 ；https://github.com/hifi-hyp/ACL-LLMPrint

Implementation:
clean-room design; no source copied.

## dreamor/llm-fingerprint

Purpose:
基于输出分布（含 JSD）与参考模型库进行黑盒模型指纹识别的工具与研究数据。

Adopt:
- **JSD（Jensen-Shannon Divergence）** 作为分布差异的透明统计量——与 LLMTrace 现有相似度方法同一梯级。
- "单 token / 短输出概率分布"作为指纹信号的粒度选择。
- 参考库（reference library）与工具分离：数据可版本化、可校验。

Do not adopt:
- 其 176 模型参考库的具体数据（不复制数据文件；如需参考集须自建并记录 provenance）。
- 任何需要下载/缓存大模型权重或向第三方服务发请求的路径。

License:
Tool: MIT / Research data: CC-BY（README 声明）。

Source:
https://github.com/dreamor/llm-fingerprint

Implementation:
clean-room design; no source copied.

## S1M0N38/llm-fingerprint

Purpose:
LLM 指纹识别项目，依赖 embedding 模型与向量数据库（ChromaDB/Qdrant）做相似度检索。

Adopt:
- "多次采样 + 分布比较"的高层思路（与 LLMmap/zing 同源）。

Do not adopt:
- **明确不采纳**：embedding 模型、ChromaDB/Qdrant 向量库、以及对应的重型依赖链——与 LLMTrace 技术梯级冲突。
- 不采纳其数据模型与运行时。

License:
MIT

Source:
https://github.com/S1M0N38/llm-fingerprint

Implementation:
clean-room design; no source copied.

## CompatBench

Purpose:
疑为针对 LLM API 兼容性/中转一致性的基准；未能定位到唯一对应的公开仓库或论文。

Adopt:
- 无（未核验到明确来源）。

Do not adopt:
- 无（同上）。

License:
Unverified — requires manual confirmation

Source:
未定位到可核验 URL；多次差异化检索命中同名但无关项目（Hadoop HCFS CompatBench、unicode-data、.NET 基准类等）。

Implementation:
clean-room design; no source copied.

## API Relay Audit

Purpose:
面向 API 中转站（relay）的审计工具，检测模型替换、降级、上下文截断等行为。

Adopt:
- **仅方法论**：把"中转站可能作弊的类别"抽象为可检测的行为信号（替换 / 量化降级 / 截断 / 静默回退）。
- 每类信号输出"证据"而非直接定罪，与 LLMTrace 的 Evidence 语义一致。
- 探测需 `temperature=0`、固定 `max_tokens`、重复多次以抵消采样噪声。

Do not adopt:
- **源码一律不复制**（AGPL-3.0-only）。
- 其实现语言、运行时与数据模型。

License:
**AGPL-3.0-only**

Source:
https://github.com/toby-bridges/api-relay-audit

Implementation:
clean-room design; no source copied. **RED LINE: source code must never be copied.**

## Cisco Model Provenance Kit

Purpose:
模型来源/谱系验证工具包，基于模型权重与架构层面的指纹（weight-level / architecture-level）。

Adopt:
- "身份验证分层"的思路：不同证据强度分级，弱证据不单独下结论。
- 将"声明身份"与"观测到的一致性"分开记录。

Do not adopt:
- **权重级 / 架构级指纹**——LLMTrace 是纯黑盒，拿不到模型权重，无法采纳。
- 其运行时与模型加载依赖。

License:
Apache-2.0

Source:
https://github.com/cisco-ai-defense/model-provenance-kit

Implementation:
clean-room design; no source copied.

## Are You Getting What You Pay For?

Purpose:
审计 LLM API 是否存在"模型替换"（Model Substitution）的研究工作（UC Berkeley），方法含 Classifier / Identity Prompting / Model Equality Testing / Benchmark / Logprobs。

Adopt:
- **Model Equality Testing** 思想：把中转站返回与"官方参考模型"做**行为等价性**比较。
- identity prompting 只作**低权重**信号（自报身份可被双向伪造，不可作为判定依据）。
- 按证据强度分层汇总，弱信号不推翻强信号。

Do not adopt:
- 其分类器训练与 logprobs 依赖（第三方中转站通常不开放 logprobs）。
- 其 harness 与外部数据集（LM Evaluation Harness 适配超出本轮范围）。

License:
MIT（配套代码 `sunblaze-ucb/llm-api-audit`）；论文 arXiv:2504.04715。

Source:
https://github.com/sunblaze-ucb/llm-api-audit ；https://arxiv.org/abs/2504.04715

Implementation:
clean-room design; no source copied.

## Rank-Based Uniformity Test

Purpose:
基于秩的均匀性检验（Rank-Based Uniformity Test），用于分布一致性/指纹匹配的统计检验。

Adopt:
- "候选模型假设下检验观测输出的秩是否均匀"的统计框架思想。
- 输出 p 值/显著性 → 与 LLMTrace 的置信语义对接（不显著即不判定）。

Do not adopt:
- 其需要 scipy/numpy 的统计实现；LLMTrace 只用 stdlib 可表达的透明统计量。
- 其对大样本的假设（LLMTrace 单次审计预算受限，需 fail-closed）。

License:
MIT（代码仓库 `xyzhu123/RUT`）；论文 arXiv:2506.06975（ICLR 2026）。

Source:
https://github.com/xyzhu123/RUT ；https://arxiv.org/abs/2506.06975

Implementation:
clean-room design; no source copied.

## tinyBenchmarks

Purpose:
用 IRT 选取 ~100 个"锚点"题目近似全量评测，并提出 IRT++ 后处理。

Adopt:
- **"锚点/固定子集"思想**（已在 v0.4-B 校准中采纳）：固定、可复现的题目子集 + 稳定身份。
- 子集选择须事前固化并 SHA 校验，避免 cherry-pick。

Do not adopt:
- **IRT 拟合本身**：稳定 IRT 参数估计需要数十个模型 × 题目级日志，LLMTrace 参考组规模（≥5 身份）不支持。

License:
MIT（代码仓库）；论文 arXiv:2402.14992。

Source:
https://github.com/felipemaiapolo/tinyBenchmarks ；https://arxiv.org/abs/2402.14992

Implementation:
clean-room design; no source copied.

## MINCE

Purpose:
Monte Carlo Informed N-sizing for Compact Evaluation——用蒙特卡洛估计最小子集规模。

Adopt:
- 无（仅作为 subset-sizing 文献记录）。

Do not adopt:
- **不采纳**：与 LLMTrace 固定 Quick Suite（SHA-256 排名、不可裁剪）前提冲突；动态子集裁剪会破坏可复现的固定套件身份。

License:
N/A (research paper; check linked repo before any code reuse)

Source:
https://arxiv.org/abs/2606.22826

Implementation:
clean-room design; no source copied.

## Micro-Benchmark Reliability

Purpose:
研究"语言模型微基准（micro-benchmarking）有多可靠"，提出 MDAD 指标；发现约 250 样本时随机采样与复杂方法表现相当。

Adopt:
- **诚实性优先**：小样本评测不声称统计显著性，显式降级为 warning + fail-closed。
- 报告 coverage（graded/total）作为测量健康度的诚实替代。

Do not adopt:
- 其复杂子集选择方法与依赖库。

License:
MIT（作者自写代码；另含 Anchor Points / tinyBenchmarks / py-irt / DPPcoresets 的适配代码，许可见其 `licenses` 目录）；论文 arXiv:2510.08730（ICLR 2026）。

Source:
https://github.com/dill-lab/micro-benchmarking-reliability ；https://arxiv.org/abs/2510.08730

Implementation:
clean-room design; no source copied.

## RouteLLM

Purpose:
LLM 路由器（router）训练与评测框架，按 query 难度在强/弱模型间路由。

Adopt:
- "路由决策应可解释、可复现、可版本化"的框架思想（对应 LLMTrace 的 Provider 路由记录）。

Do not adopt:
- 其训练/推理栈与外部模型、GPU 依赖。

License:
Apache-2.0

Source:
https://github.com/lm-sys/RouteLLM

Implementation:
clean-room design; no source copied.

## RouterEval

Purpose:
LLM 路由器评测基准（EMNLP），评估路由性能。

Adopt:
- "多路由器 × 多基准"的横向对比组织方式，用于设计 LLMTrace 的路由记录字段。

Do not adopt:
- 其数据集与评测 harness。

License:
MIT

Source:
https://github.com/MilkThink-Lab/RouterEval

Implementation:
clean-room design; no source copied.

## LLMRouterBench

Purpose:
LLM 路由基准（ACL 2026 Findings）。

Adopt:
- 路由评测的指标组织思路。

Do not adopt:
- 其数据集、harness 与运行时。

License:
Unverified — requires manual confirmation（仓库 `main` 分支**未见 LICENSE 文件**，访问 `blob/main/LICENSE` 返回 404）；论文 arXiv:2601.07206 为 CC BY 4.0。

Source:
https://github.com/ynulihao/LLMRouterBench ；https://arxiv.org/abs/2601.07206

Implementation:
clean-room design; no source copied.

## Inspect AI

Purpose:
LLM 评测框架（UK AI Safety Institute），提供 Solver / Scorer / Dataset / Task 抽象与日志查看。

Adopt:
- **Solver / Scorer / Dataset 解耦**思想：评测逻辑与评分逻辑分离，便于组合与复用。
- 评测过程产生结构化日志（可追溯），与 LLMTrace 的 Evidence/Artifact 闭合链同构。

Do not adopt:
- 其运行时（引入即等于新增依赖）与其 task 数据模型（破坏 LLMTrace `frozen` + `extra=forbid` 约束）。

License:
MIT

Source:
https://github.com/UKGovernmentBEIS/inspect_ai

Implementation:
clean-room design; no source copied.

## EvalScope

Purpose:
ModelScope 生态的评测框架，覆盖多后端、多数据集。

Adopt:
- "统一评测入口 + 可插拔后端适配器"的组织思想（对应 LLMTrace Provider 抽象）。

Do not adopt:
- 其重型依赖链与数据集缓存机制。

License:
Apache-2.0

Source:
https://github.com/modelscope/evalscope

Implementation:
clean-room design; no source copied.

## OpenCompass

Purpose:
开源 LLM 评测平台（上海 AI Lab），支持大量数据集与分布式评测。

Adopt:
- 数据集/模型/评测器三层解耦与配置驱动思想。

Do not adopt:
- 其分布式运行时与庞大依赖（超出 local-first 单机范围）。

License:
Apache-2.0

Source:
https://github.com/open-compass/opencompass

Implementation:
clean-room design; no source copied.

## LiveBench

Purpose:
抗污染（contamination-free）的持续更新评测基准。

Adopt:
- **"评测集需防污染、需版本化"**的思想：固定套件身份 + 内容 SHA 已在 LLMTrace 落地。
- 定期更新题目以对抗训练集泄漏的思路（记录为未来方向）。

Do not adopt:
- 其题目数据与评分 harness。

License:
Apache-2.0

Source:
https://github.com/LiveBench/LiveBench

Implementation:
clean-room design; no source copied.

## OpenAI Evals

Purpose:
OpenAI 的评测框架与数据集集合。

Adopt:
- "Eval = 数据集 + 评分函数 + 记录"的最小结构。
- 数据来源与许可需逐集标注的做法（LLMTrace 的 reference/suite 也须记录 provenance）。

Do not adopt:
- 其 registry/运行时与数据集本体。

License:
MIT（仓库主许可；各数据集可能另有单独许可，复制数据前须逐集核验）。

Source:
https://github.com/openai/evals

Implementation:
clean-room design; no source copied.

## Promptfoo

Purpose:
LLM 测试/红队框架：test case / assertion / threshold / provider-independent result。

Adopt:
- **一个 item 由多个独立断言（assertion）分析，逐项产出结构化结果**——对应 LLMTrace 的多 detector → signals 设计。
- 阈值集中到 policy，不散落在 if 语句。

Do not adopt:
- 其 TypeScript/Node 运行时与其数据模型。

License:
MIT

Source:
https://github.com/promptfoo/promptfoo

Implementation:
clean-room design; no source copied.

## Evidently

Purpose:
数据/模型漂移与质量监控：Reference / Current / Metric / Test / Threshold / Result。

Adopt:
- **Reference-vs-Current 成对比较 + 版本化 Threshold policy**（LLMTrace 的 BehaviorDriftPolicy 已采纳此思想）。
- 阈值不散落、结果结构化。

Do not adopt:
- 其 Python 依赖栈与数据模型。

License:
Apache-2.0

Source:
https://github.com/evidentlyai/evidently

Implementation:
clean-room design; no source copied.

## Langfuse

Purpose:
LLM 可观测平台：Trace / Observation / Generation / Score 层级。

Adopt:
- **中央观测层级**思想：HTTP Evidence = Observation、Benchmark Item = Generation、Capability/Finding = Score（LLMTrace `EvidenceRecorder` 已对齐）。

Do not adopt:
- 其服务端运行时、Web 前端与数据库（LLMTrace 用本地 sqlite / 文件工件）。

License:
MIT Expat（**仅限** `ee/`、`web/src/ee/`、`worker/src/ee/` 目录**之外**的内容）；上述 `ee/` 目录由 `ee/LICENSE` 单独定义（商业许可）。

Source:
https://github.com/langfuse/langfuse

Implementation:
clean-room design; no source copied.

## vLLM Bench

Purpose:
vLLM 提供的推理性能基准（吞吐、延迟、TTFT 等）。

Adopt:
- 性能指标口径（TTFT / 吞吐 / 延迟分位数）作为 LLMTrace 请求证据字段的参考。

Do not adopt:
- vLLM 运行时与 GPU 依赖。

License:
Apache-2.0（复用 vLLM 仓库许可）

Source:
https://github.com/vllm-project/vllm

Implementation:
clean-room design; no source copied.

## LLMPerf

Purpose:
LLM 推理性能压测工具（官方标注已 archive）。

Adopt:
- 负载测试的指标定义（TTFT、inter-token latency、并发）与结果聚合方式。

Do not adopt:
- 其运行时与压测并发框架（LLMTrace 为单机审计，不做大规模压测）。

License:
Apache-2.0

Source:
https://github.com/ray-project/llmperf

Implementation:
clean-room design; no source copied.

## Inferock Bench

Purpose:
面向模型/供应商正确性（correctness）的基准与度量（含 spec 与 measure 包）。

Adopt:
- **仅方法论**："正确性 vs 性能"分组度量的组织思路。

Do not adopt:
- **源码一律不复制**。
- 其运行时、spec 与度量实现。

License:
**FSL-1.1-ALv2**（`apps/inferock-bench`，Functional Source License，source-available，后续转为 Apache-2.0）；其余目录：`packages/measure` = Apache-2.0、`spec` = CC-BY-4.0。

Source:
https://github.com/inferock/inferock-bench

Implementation:
clean-room design; no source copied. **RED LINE: main repo is source-available/FSL; source code must never be copied.**

## MetaReason Core

Purpose:
面向推理/元推理的框架，使用 PyMC 贝叶斯、拉丁超立方采样、SQLite 存储与 CLI + Python API。

Adopt:
- **CLI + Python API 双入口**与 **SQLite stdlib 持久化**的组织方式（与 LLMTrace 技术梯级吻合）。
- 结果落库可复现、可追溯。

Do not adopt:
- PyMC 贝叶斯采样与 LHS（需重型统计依赖，LLMTrace 只用 stdlib 透明统计）。

License:
MIT

Source:
https://github.com/metareason-ai/metareason-core

Implementation:
clean-room design; no source copied.

## Your Agent Is Mine

Purpose:
研究性工作，含一个 FastAPI 代理（proxy）用于观测/审计 agent 与 API 之间的流量（ACM CCS 2026）。

Adopt:
- **"代理层观测"**思想：把请求/响应在中间层完整留证（LLMTrace 的 HTTPEvidence 已采用此模式）。
- 明确记录代理本身的信任边界。

Do not adopt:
- 其研究代码与其代理实现（未见独立公开代码仓库）。
- 任何需要持久化凭据的设计。

License:
N/A (research paper; check linked repo before any code reuse)
（论文 arXiv:2604.08407，标注 CC BY-NC-ND 4.0；NC/ND 限制使其即便有代码也不可商用/不可改编。）

Source:
https://arxiv.org/abs/2604.08407

Implementation:
clean-room design; no source copied.

---

## 本轮 clean-room 实现审计（Task 60）

审计范围（v0.6 本轮新增）：

- `src/llmtrace/fingerprint/`：`models.py` / `suite.py` / `normalizers.py` / `distribution.py` /
  `aggregation.py` / `distance.py` / `matcher.py` / `reference.py` / `reference_set.py` /
  `repository.py` / `policy.py` / `validation.py` / `temporal.py` / `routing.py` /
  `executor.py` / `discovery.py` / `runtime.py` / `resources/fingerprint_v1.json`
- `src/llmtrace/analysis/{confidence_v2,performance,resolution}.py`、`src/llmtrace/reporting/fingerprint_report.py`
- 配套测试：`tests/fingerprint/`、`tests/cli/test_cli_fingerprint*.py`、`tests/analysis/`、
  `tests/integration/test_fingerprint_e2e.py`

审计方法与结论：

1. 对 `src/llmtrace/` 全目录检索 `copyright` / `SPDX` / `licensed under` / `AGPL` / `FSL-` /
   `adapted from` / `copied from` / `derived from` 等标识：**唯一命中为 LLMTrace 自身对
   `openai/gsm8k`（MIT）数据集来源的说明**，与本轮新增代码无关；新增代码中不含任何外部项目的
   版权头、许可声明或供应商文件名。
2. 逐项确认实现来源（全部为 clean-room，仅使用 Python stdlib）：

   | 能力 | 实现位置 | 依赖 |
   |---|---|---|
   | JSD / KL 散度 | `fingerprint/distance.py` | stdlib `math` |
   | 选项归一化（NFKC + strip + casefold 整段匹配） | `fingerprint/normalizers.py` | stdlib `unicodedata` |
   | 离散分布与 `__INVALID__` 分母 | `fingerprint/distribution.py` | stdlib `collections.Counter` |
   | held-out 验证与 threshold 选择 | `fingerprint/validation.py` | 本项目自有 `FingerprintMatcher` |
   | decision policy（自哈希、FAR 上限 fail closed） | `fingerprint/policy.py` | 本项目 `utilities/hashing.py` |
   | routing v2 强/支持两层信号 | `fingerprint/routing.py` | stdlib `statistics` |
   | 变异系数等离散度统计 | `fingerprint/routing.py` | stdlib `statistics.pstdev` |

   未引入 numpy / scipy / sklearn / sentence-transformers / embedding 数据库 / 新 HTTP client
   （Rule 5 技术梯级）。
3. 红线项目（API Relay Audit = AGPL-3.0-only、Inferock Bench = FSL-1.1-ALv2）在本轮实现中
   **零引用**：既没有代码，也没有数据模型、字段命名或文件名借用。
4. 许可证核验：本轮实现**不依赖**任何外部项目的许可证。凡未能在仓库中核验到明确 `LICENSE`
   的来源，均记为 **license not relied upon for implementation**（不得猜测其许可类型）。
5. attribution：本轮采纳的方法论（JSD 分布匹配、held-out 验证、Top-K 参考比对、RUT 式
   统计检验思路、LLMPrint 式更强验证）已在本文档上述条目中保留点名致谢，并在
   `docs/methodology/fingerprinting.md` 的 "Future research" 段落中再次标注来源。
