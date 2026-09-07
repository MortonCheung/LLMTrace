# Open-Source Design Influences

LLMTrace 在**不引入任何 runtime 依赖**的前提下，从多个成熟开源项目中吸收了解耦思想。
下表记录"借鉴了什么、如何落地、是否复制代码"。

本轮原则：只吸收架构思想，自行实现轻量 Python 版本；不安装、不照搬数据模型、
不引入 TypeScript/Go runtime；若未来复制任何源码，必须先行核对许可证。

## v0.3-D — Behavior Drift

| Project | Relevant concept | How LLMTrace adapts it | Code copied | License note |
|---|---|---|---|---|
| [projectdiscovery/nuclei](https://github.com/projectdiscovery/nuclei) | Template / Matcher / Extractor / Finding 解耦 | 检测规则（Detector）与执行器（BehaviorDriftEngine）解耦：Detector 只产出局部 `BehaviorSignal`，Engine 负责对齐与汇总 | No | MIT（仅思想借鉴，未复制） |
| [NVIDIA/garak](https://github.com/NVIDIA/garak) | Probe / Detector / Evaluator / Harness 拆分 | `BehaviorDetector` Protocol 插件化：Outcome / Status / Output / Operational 四类 detector 独立、无副作用、不决定最终 Drift Level | No | Apache-2.0（仅思想借鉴，未复制） |
| [promptfoo/promptfoo](https://github.com/promptfoo/promptfoo) | test case / assertion / threshold / provider-independent result | 一个 item 观测由多个独立 detector 分析，逐项生成 `ItemDriftResult.signals`；阈值集中进 `BehaviorDriftPolicy` | No | MIT（仅思想借鉴，未复制） |
| [evidentlyai/evidently](https://github.com/evidently/evidently) | Reference / Current / Metric / Test / Threshold / Result | Reference-vs-Current 漂移由版本化 `BehaviorDriftPolicy` 控制（阈值不散落 if 语句），baseline/current 两 snapshot 成对比较 | No | Apache-2.0（仅思想借鉴，未复制） |

## v0.3-E — Unified Execution & Artifact Foundation

| Project | Relevant concept | How LLMTrace adapts it | Code copied | Runtime dep | License note |
|---|---|---|---|---|---|
| [mlflow/mlflow](https://github.com/mlflow/mlflow) | Run / Param / Metric / Artifact / Tag | `UnifiedRunResult` = Run、执行配置 = Param、能力/请求指标 = Metric、报告与快照 = Artifact、target 元数据 = Tag；append-only `RunArtifactRepository` 借鉴其 run 目录思想 | No | No | Apache-2.0（仅思想借鉴） |
| [langfuse/langfuse](https://github.com/langfuse/langfuse) | Trace / Observation / Generation / Score | 统一执行的中央观测层级 → `EvidenceRecorder`：HTTP Evidence = Observation、Benchmark Item = Generation、Capability/Finding = Score | No | No | MIT（仅思想借鉴） |
| [helicone/helicone](https://github.com/Helicone/helicone) | request-level latency / tokens / model / provider / cost | 校验 `HTTPEvidence` 的 request 观测字段（latency / tokens / model / provider / cost），request budget 对齐其 request 计量 | No | No | Apache-2.0（仅思想借鉴） |
| [BerriAI/litellm](https://github.com/BerriAI/litellm) | Provider abstraction / Model metadata / Cost registry | **仅研究**其 Provider 抽象与成本注册表；本轮不引入 runtime 依赖，Target API 必须由 LLMTrace 原生 Provider 抓完整 Evidence。LiteLLM 正式 ReferenceProvider 留待 v0.4 | No | No | MIT（仅思想借鉴） |

## v0.4-B — Reference Calibration（方法论研究，非代码借鉴）

校准设计前做的一次轻量方法论调研。目的不是复制实现，而是确认
piecewise anchor 选择没有明显统计学错误、且相对 IRT 有可辩护的依据。

| 来源 | 相关概念 | LLMTrace 采纳/不采纳 | 采纳方式 |
|---|---|---|---|
| [Artificial Analysis Intelligence Index](https://artificialanalysis.ai/methodology/intelligence-benchmarking) | 加权综合指数、方法论版本化、重复运行置信区间、"指数分是相对比较工具而非绝对度量" | **采纳**：版本化 CalibrationPolicy（id+version 贯穿 provenance）、指数语义（0–100 是相对校准宇宙的坐标）、能力/速度/成本分开报告 | 仅思想；AA 用 judge panel + Elo，LLMTrace 用确定性 item-level 评分 |
| [tinyBenchmarks (arXiv:2402.14992)](https://arxiv.org/abs/2402.14992) | IRT anchor 点选取 100 题近似全量评测、IRT++ 后处理 | **不采纳 IRT，采纳"锚点"思想**：IRT 拟合需要大量模型 × 题目级日志（数十个校准模型），LLMTrace 参考组最小 5 个身份，样本量不支持稳定 IRT 参数估计；piecewise anchor 在小参考组下方差可控且可解释 | Quick Suite 固定 32 题与 anchor 校准正交 |
| [lm-evaluation-harness stderr/bootstrap 实践](https://github.com/EleutherAI/lm-evaluation-harness) | 逐题 stderr、bootstrap 置信区间 | **部分采纳（未来方向）**：32 题 × 4 维的 bootstrap CI 过宽，本轮把测量退化显式降级为 warning + fail-closed，不声称统计显著性；报告 coverage 指标（graded/total）作为诚实性替代 | v0.5+ 可在 Quick Suite 重复运行上引入 |
| [Growing Pains: fixed parameter calibration (arXiv:2604.12843)](https://arxiv.org/abs/2604.12843) | 固定锚点集校准新基准以保持跨期可比（IRT 固参法） | **采纳"锚点保持可比"思想**：ReferenceSet content SHA + CalibrationPolicy 版本共同构成坐标系；换参考组 = 换坐标系，分数不可跨组比较（文档语义红线） | 其发现"精确近似需要数十个参考模型"印证了我们对 v1 精度的保守定位 |
| [MINCE (arXiv:2606.22826)](https://arxiv.org/abs/2606.22826) | Monte Carlo 最小子集规模估计 | **不采纳**：与 LLMTrace 的固定 Quick Suite（SHA-256 排名、不可 cherry-pick）前提冲突；子集裁剪会破坏可复现的固定套件身份 | 仅作为 subset-sizing 文献记录 |

结论：v1 选择 piecewise linear anchor 而非 IRT 的依据——参考组规模（≥5 身份）
远低于 IRT 稳定拟合需求；piecewise 映射单调、可解释、fail-closed 条件明确；
后续参考组扩大后可评估 IRT（新 policy 版本，不改动 v1）。

## 为什么不是复制

- 四个项目各绑定自己的运行时与数据模型，直接引入会破坏 LLMTrace 的
  "评分/Evidence/报告"闭合链与 `frozen` + `extra=forbid` 领域约束。
- LLMTrace 需要的是 `BehaviorItemKey = (task_id, source_sample_id, input_sha256)`
  这种自有稳定身份，而非外部项目的 test case 身份。
- Drift Level 必须与 LLMTrace 已有的 `UNCALIBRATED` / fail-closed 语义一致，
  外部项目的阈值语义（尤其置信区间、统计检验）不在本轮范围。

## 语义红线（对外部思想的本地位约束）

- Output hash 变化 ≠ 模型偷换；只算"行为变化观测"。
- FAILURE 的强制 0 分 ≠ 能力下降到 0；只算状态变化。
- 不同 generation config / suite / policy 下**不可比较**，必须 fail closed。
