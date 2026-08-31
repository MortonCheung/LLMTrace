# Open-Source Design Influences

LLMTrace v0.3-D 的 Behavior Drift 架构在**不引入任何 runtime 依赖**的前提下，
从四个成熟开源项目中吸收了解耦思想。下表记录"借鉴了什么、如何落地、是否复制代码"。

本轮原则：只吸收架构思想，自行实现轻量 Python 版本；不安装、不照搬数据模型、
不引入 TypeScript/Go runtime；若未来复制任何源码，必须先行核对许可证。

| Project | Relevant concept | How LLMTrace adapts it | Code copied | License note |
|---|---|---|---|---|
| [projectdiscovery/nuclei](https://github.com/projectdiscovery/nuclei) | Template / Matcher / Extractor / Finding 解耦 | 检测规则（Detector）与执行器（BehaviorDriftEngine）解耦：Detector 只产出局部 `BehaviorSignal`，Engine 负责对齐与汇总 | No | MIT（仅思想借鉴，未复制） |
| [NVIDIA/garak](https://github.com/NVIDIA/garak) | Probe / Detector / Evaluator / Harness 拆分 | `BehaviorDetector` Protocol 插件化：Outcome / Status / Output / Operational 四类 detector 独立、无副作用、不决定最终 Drift Level | No | Apache-2.0（仅思想借鉴，未复制） |
| [promptfoo/promptfoo](https://github.com/promptfoo/promptfoo) | test case / assertion / threshold / provider-independent result | 一个 item 观测由多个独立 detector 分析，逐项生成 `ItemDriftResult.signals`；阈值集中进 `BehaviorDriftPolicy` | No | MIT（仅思想借鉴，未复制） |
| [evidentlyai/evidently](https://github.com/evidently/evidently) | Reference / Current / Metric / Test / Threshold / Result | Reference-vs-Current 漂移由版本化 `BehaviorDriftPolicy` 控制（阈值不散落 if 语句），baseline/current 两 snapshot 成对比较 | No | Apache-2.0（仅思想借鉴，未复制） |

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
