# Routing Methodology — 路由不一致证据（v0.6，Experimental）

本文件说明 LLMTrace 的 **routing 稳定性判定**（v1 与 v2）实际能说什么、不能说什么。
实现位于 `src/llmtrace/analysis/routing.py`（v1，保留不动）与
`src/llmtrace/fingerprint/routing.py`（v2，`assess_routing_v2`）。

---

## 1. 三个层次：只有前两层可以声明

| 层次 | 内容 | LLMTrace 是否支持 |
|---|---|---|
| **1. observed inconsistency** | 在本次样本中**观察到**服务端给出的 model 标识/行为分布不稳定 | ✅ 支持 |
| **2. possible mixed routing** | 基于强证据，判断"混合路由**是可能的**" | ✅ 支持 |
| **3. proven routing mixture** | 声称**已证明**存在按某比例的混合路由（例如 "70% A / 30% B"） | ❌ **不支持，禁止声明** |

第 3 层需要 LLMTrace 目前**不具备**的东西：对路由概率的统计识别、对上游身份的独立确认、
以及对"多来源流量按比例混合"的可识别性假设。在缺少这些前提时给出比例或"已证明混合"
属于**无依据的强主张**，因此实现层从文本到模型都禁止这类输出：

- 只输出既有四个标签（`Stable` / `Mostly Stable` / `Suspicious` / `Insufficient Data`），
  不输出任何百分比混合比例；
- 允许的措辞只有 `routing inconsistency observed` / `mixed routing is possible`
  这类**行为层**描述；
- `RoutingAssessmentV2` 携带 `experimental = True`，明示这是实验性标签而非统计证明。

---

## 2. v1 与 v2 的关系

- **v1**（`analysis/routing.py`）：只看服务端自报的 model identifier 与失败率。
  高失败率在 v1 中可以直接把等级推高。
- **v2**（`fingerprint/routing.py`）：沿用同一组标签，但把证据分成**强 / 支持**两层强度，
  并接入指纹的时间维证据。**v1 实现保留不动**，v2 是 v0.6 新增的判定路径。
- v2 的关键降级：**失败率与延迟/token 离散度在 v2 中只是支持证据**，
  不再单独把等级推为 `Suspicious`。原因是"端点不稳定"与"路由到了不同模型"
  是两件不同的事——限流、超时、过载都会造成高失败率。

---

## 3. v2 证据分级

### 3.1 强证据（strong）——足以单独支持 Suspicious

| 信号 ID | 触发条件 |
|---|---|
| `multiple_stable_response_model_identifiers` | 样本中出现 ≥ 2 个不同的 `response_model` 标识，且**没有单一主导**（最大占比 < `minimum_dominant_ratio`，默认 `0.90`）。只有一个"游离标识"不算。 |
| `reference_match_switches_between_temporal_windows` | 在 ≥ 2 个时间窗上各自做参考匹配，**最近参考身份或匹配状态发生切换**。要求存在**已验证** policy。 |
| `temporal_divergence_exceeds_validated_reference_baseline` | 候选自身的时间散度**超过**已验证 policy 携带的参考基线（`temporal_divergence_baseline`）。要求存在**已验证** policy 且其携带非空基线。 |

后两条受 **Rule 2 门禁**：没有 validated `FingerprintDecisionPolicy` 时，
所有 temporal 强证据**一律不成立**，并作为 limitation 记录在评估结果里。

### 3.2 支持证据（supporting）——只能与其它观察一起支持判断

| 信号 ID | 触发条件 |
|---|---|
| `latency_dispersion` | 延迟的变异系数（population stdev / mean）> `dispersion_coefficient_threshold`（默认 `0.50`） |
| `token_dispersion` | 输出 token 数的变异系数 > 同一阈值 |
| `provider_failure` | provider 失败率 ≥ `degraded_failure_ratio_threshold`（默认 `0.25`） |

**支持证据单独出现时不得得出 "mixed routing detected"。** 在 v2 的判定顺序中，
仅命中支持证据（无强证据）时等级最高只能到 `Mostly Stable`，并在 `reasons` 中
列出具体命中的支持信号，而不是升级为 `Suspicious`。

### 3.3 判定顺序（fail closed）

```text
1. Insufficient Data   样本数 < minimum_samples（默认 8），或服务端从未报告任何 model 标识
2. Suspicious          至少一条强证据被观察到
3. Mostly Stable       无强证据，但：存在支持证据 / 存在多于一个标识 / 没有可用的 temporal fingerprint
4. Stable              无强证据、无支持证据、单一 model 标识，且 temporal fingerprint 可用
```

`Stable` 需要"正面确认时间稳定性"，因此**缺少 temporal fingerprint 时不会给 Stable**，
只会给 `Mostly Stable` 并在 `reasons` / `limitations` 中说明原因。

版本化阈值（`RoutingV2Policy`，`policy_id = llmtrace-routing-v2`，`policy_version = 2.0.0`）：
`minimum_samples = 8`、`minimum_dominant_ratio = 0.9`、
`degraded_failure_ratio_threshold = 0.25`、`dispersion_coefficient_threshold = 0.5`。
**改阈值必须换版本**（与 v1 / `ConfidencePolicy` 同纪律）。

---

## 4. Routing Suspicious 是审计结论，不是 Run Failed

这是本方法论最容易被误读的一点，必须写清：

- `Suspicious` 是**证据层的判定标签**，它描述"观察到了与稳定单一路由不一致的迹象"。
- 它**不**改变 run 的执行状态：一次 `run --verify-model` 在 routing 报 `Suspicious` 时，
  **run status 仍然是 `COMPLETED`**。
- 它只写入报告的 `fingerprint.routing` 段落
  （含 `level` / `distinct_response_models` / `dominant_model_ratio`），
  与 capability / calibration 的既有语义**互不干扰**（Rule 1：不得破坏既有能力与校准语义）。

**Scenario D 的语义（端到端测试 `tests/integration/test_fingerprint_e2e.py`）**：
mock 端点让 `response_model` 在两个标识之间交替，路由评估给出 `Suspicious`，
同时断言 `result.status is UnifiedRunStatus.COMPLETED`——
即"路由可疑"与"这次审计执行失败"是两件独立的事：

- `COMPLETED` / `PARTIAL` / `FAILED` 描述的是**本次测量是否健康完成**；
- `Suspicious` 描述的是**在健康完成的测量中观察到了什么**。

因此报告与 CLI 输出**不得**把 routing 可疑渲染成"运行失败"，也不得据此把 run 标记为
`FAILED` 或取消报告生成。

---

## 5. 与外部研究的关系（attribution）

v2 的"强 / 支持"两层证据设计只借鉴公开的方法论思路（路由/混合模型的统计识别问题），
**不复制任何外部源码**（含 RouteLLM、RouterEval、LLMRouterBench 等），
不引入其 runtime 与数据模型。逐项说明与许可边界见
[`docs/research/external-influences.md`](../research/external-influences.md)；
未能核验许可的来源记为 **license not relied upon for implementation**。

## 6. Future research（仅登记，不实现）

真正走到"proven routing mixture"（第 3 层）需要：混合比例的可识别性分析、
对上游身份的独立确认、以及跨时间窗的统计检验。这些属于后续研究，
与 [`fingerprinting.md`](./fingerprinting.md#8-future-research仅登记不在本轮实现) 中的
RUT / LLMPrint 式验证、open-set verification、mixture 估计同属一个方向，
本轮**不实现**，也不输出任何比例数字。
