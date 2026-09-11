# Fingerprinting Methodology — 行为身份证据（v0.6，Experimental）

本文件说明 v0.6 「Model Identity Evidence / Fingerprint Foundation」的完整方法与语义边界。
它描述的是 **LLMTrace 当前实际实现**的行为，不是研究计划；实现代码位于
`src/llmtrace/fingerprint/`。

> **一句话边界**：LLMTrace 产出的是 **behavioral identity evidence（行为身份证据）**——
> "本次观测到的行为分布，是否与**受信任指纹参考集中被声称身份**的参考分布一致"。
> 它**不是**身份证明，**不是**密码学证明，也不产生"真实上游模型是 X"这类结论。

---

## 1. 完整数据链

一次 `llmtrace run --verify-model` 的身份证据链如下（每一步都可以在 artifact 中被复核）：

```text
probe（低熵分类探测项）
  → repeated samples（重复轮次采样，按轮交错）
  → normalization（严格归一化到某个选项，或 __INVALID__）
  → categorical distribution（每个 probe 一个离散分布，support = choices + __INVALID__）
  → per-probe JSD（本次分布 vs 参考分布）
  → weighted aggregate distance（按 probe 权重聚合）
  → trusted reference matching（与参考集中每个 identity 比对并排序）
  → Top-K（距离升序的行为最接近参考）
  → held-out validation（离线：leave-one-capture-out 验证参考集）
  → decision policy（离线：只有通过验证才产生 threshold）
  → claim consistency（在线：与"被声称身份"的参考距离对照 threshold）
```

### 1.1 时间轴：离线建信任链 vs 在线产生观测

链条里有两类活动，**不要混为一谈**：

| 阶段 | 何时发生 | 是否发送 HTTP | 产物 |
|---|---|---|---|
| probe → Top-K | 每次 `run --verify-model` | 是（`probe_count × repetitions` 次） | `fingerprint_snapshot.json` / `fingerprint_match.json` |
| held-out validation → decision policy | `llmtrace fingerprint validate`（离线） | 否 | `FingerprintDecisionPolicy` |
| claim consistency | 每次 `run --verify-model` 的报告阶段 | 否 | `fingerprint_verification.json` |

### 1.2 逐步说明

1. **probe**：套件 `fingerprint_v1.json`（`suite_id = llmtrace-fingerprint-categorical`）声明 6 个
   低熵分类探测项（symbol / direction / prime / season / weekday / color-choice），每项 2–4 个互不重复的
   选项，`temperature = 1.0`、`max_output_tokens = 8`、默认 `weight = 1.0`。套件内容身份由
   `content_sha256` 自校验；更换套件内容即产生新的套件身份。
2. **repeated samples**：`--fingerprint-profile` 决定重复轮数（`quick = 4` / `standard = 8` /
   `research = 16`）。执行时**按轮次交错**（round-robin，每轮内 probe 顺序按 seed 打乱），
   而不是"probe A 跑 N 次、再跑 probe B N 次"——避免时间漂移或服务端 routing 变化被误读成 probe 差异。
3. **normalization**：第一版刻意严格，只有"整段文本经 NFKC + strip + casefold 后**恰好等于**某个选项"
   才算命中（`llmtrace-exact-choice` v1.0.0）。**不做子串匹配**：
   `"Maybe circle, but square also works"` 不会被判成选了 circle。无法判定时落 `__INVALID__`，
   **绝不猜最接近的选项**。未知归一化策略 fail closed。
4. **categorical distribution**：每个 probe 的输出分布，`support = choices + "__INVALID__"`（顺序固定），
   `sample_count` 恒等于该 probe 真实发出的请求数。多次 capture 的聚合分布取**各次 capture 概率的算术平均**
   （每次 capture 等权，不按样本总数加权），避免一次超大 capture 支配全部历史。
5. **per-probe JSD**：对每个 probe 计算候选分布与参考分布的 Jensen–Shannon 散度（2 为底，取值 `[0, 1]`）。
   JSD 只在 **support 完全一致、两侧都有样本、概率已归一化**时才计算；否则该 probe 以显式 `note`
   记为 *not comparable*，**不静默跳过、不缩小分母**。
6. **weighted aggregate distance**：`sum(weight × per_probe_jsd) / sum(active_weight)`，
   只对 comparable probe 聚合。可比 probe 数低于 decision policy 的
   `minimum_comparable_probes` 时 **fail closed**，拒绝给出聚合距离。
7. **trusted reference matching**：与参考集中**每一个** identity 的聚合分布各算一次距离，
   产出排序结果。参考条目必须与本次套件身份一致（`suite_id` / `suite_version` / `content_sha256`），
   来自其他套件的参考**永不参与排序**。
8. **Top-K**：按 `(distance, provider_id, model_id)` 升序确定排序；`similarity = 1 - distance`
   只是同一数字的可读写法，**不是概率、不是置信度、不是准确率**。
9. **held-out validation**：`llmtrace fingerprint validate` 对参考集做 leave-one-capture-out——
   每次留出一个 capture 当 held-out，参考库为"集合减去该 capture"，记录与自身 identity 的距离、
   最近 impostor 距离、top-1 / top-3 是否正确。最低条件：**≥ 5 个不同 identity，每个 identity ≥ 2 次
   独立 capture**；不满足则 `validated = False`，**不生成 threshold，也不发布任何准确率数字**。
10. **decision policy**：在 `false_accept_rate <= --max-far`（默认 `0.01`）的候选阈值中**最大化 TPR**
    选出 `distance_threshold`；找不到满足约束的候选同样 `validated = False`。
    policy 是 immutable / self-hashed 的：`validated == False` 时**在模型层禁止**携带
    `distance_threshold` / `true_positive_rate` / `false_accept_rate` / temporal baseline，
    杜绝"手工写一个好看的数字"。
11. **claim consistency**：只有同时满足下列三项，才允许输出
    `BEHAVIOR_CONSISTENT_WITH_CLAIM` / `BEHAVIOR_INCONSISTENT_WITH_CLAIM`：
    - **validated** `FingerprintDecisionPolicy`（含 threshold）；
    - 参考集中存在**被声称身份**的参考条目（取该标签下**最小**参考距离，
      即"与声称标签最接近的那次采集"——偏向"一致"方向的保守选择）；
    - 可比证据足够（comparable probe 数达到 policy 下限）。

    任一不满足时，结论等级只能是 `RANKED_ONLY`（有排序、无判定）或 `INCONCLUSIVE`（没有可比参考）。

---

## 2. 为什么 Capability Benchmark 与 Fingerprint Probe 不能合并

两者回答的是**不同的问题**，且各自的失败模式会互相污染：

| | Capability Benchmark | Fingerprint Probe |
|---|---|---|
| 问的问题 | "这个端点**能做什么**"（能力坐标） | "这个端点**的行为分布像谁**"（claim consistency） |
| 输出 | 0–100 校准能力分、分维度分、与参考模型差距 | Top-K 行为距离、claim consistency |
| 对**正确答案**敏感 | 是（判分依赖正确性） | 否（只看选项分布的稳定性） |
| 对**版本/采样/系统提示**敏感 | 弱（换版本能力可能仍相近） | 强（分布本身就会变） |
| 失败语义 | Provider Failure 会污染能力结论 | `__INVALID__` 是分布的一个取值 |
| 证据身份 | ReferenceSet / CalibrationPolicy | FingerprintSuite / FingerprintReferenceSet / FingerprintDecisionPolicy |

合并会同时破坏两边：

- **能力分不能当身份证据**：能力分是可被"降级但同类"端点拉近的相对坐标（同族模型、不同量化、
  提示包装、缓存都可能改变分数），把它当身份判据等于把"能力相近"误读成"就是同一个模型"。
- **行为分布不能当能力结论**：指纹 probe 是低熵选择题，答对与否与模型能力无关；
  把它的分布差异写进能力分，会让能力分失去可解释的校准语义。
- **域不可交叉污染**（实现约束）：fingerprint 域**不向** `ReferenceSet` / `ReferenceSnapshot` /
  `CapabilityProfile` / `CalibrationPolicy` 写入任何字段；两个域只在 artifact 层并列出现。
  被测端点当次的 capture 标记为 `candidate_capture`，**永不进入**受信任指纹集——否则参考库会被
  被测端点自己的行为污染，形成循环论证。

---

## 3. `__INVALID__` 是证据，不是应该被删除的数据

- 归一化失败、HTTP 非 2xx、异常、空输出——**全部**记为 `__INVALID__`，并且 `sample_count` 仍然
  等于真实发出的请求数。分母**永不缩小**。
- `__INVALID__` 是分布 support 中的**一个正常取值**，参与 JSD 计算。它是"这个端点在这些请求上
  不可解析/失败"的**观测事实**。
- 删除它会造成两类错误结论：(a) 分母被缩小，剩余选项概率被虚高，分布看起来"更干净"；
  (b) 高失败率的端点在行为距离上反而"看起来很像"参考——即失败被伪装成一致。
- 因此 `ProbeDistribution` 在模型层强制：support 必须以 `__INVALID__` 结尾，
  `invalid_count` 必须等于该取值计数，`counts` 之和必须等于 `sample_count`。
  `valid=False` 的观测**必须**携带 `__INVALID__`，不允许"无效却记成某个选项"。
- 指纹执行器不做 silent retry：重试会改变真实请求数，从而破坏 `planned_requests` 契约与可复核性。

---

## 4. JSD 与"距离越小 = 行为分布越接近"

- 选择 **Jensen–Shannon 散度**（2 为底）的原因：它对两个分布**对称**、有界 `[0, 1]`、
  在 support 一致时定义良好，且只用 Python `math` 即可实现（不引入 numpy / scipy / sklearn）。
- 语义：**JSD 越小 ⇒ 候选与参考在逐选项概率上越接近**；`0` 表示逐选项概率完全相同；
  `1` 表示在给定 support 上完全分离。
- JSD **不是**概率、不是"是同一个模型的概率"、不是置信度，也不可与"准确率"互换。
- 聚合用的 `similarity = 1 - distance` 只是可读写法，便于报告展示排序，
  **不改变它仍是距离这一事实**。

---

## 5. 为什么 Top-1 不能生成 identity verdict

Top-1 回答的是"**在参考集中，哪个参考的行为分布离本次观测最近**"。它**不**回答
"这个端点是不是该模型"。原因：

1. **参考集是有限的、人工挑选的**：Top-1 只是在"已登记的候选"中最接近的一个，
   不是开放世界中的身份判定（open-set 问题未解决）。
2. **没有验证过的判定规则**：没有 threshold 就没有边界，"更近"只是相对排序。
3. **可以被规避**：低熵选择题的行为分布可被提示包装、采样参数、后处理改变；
   行为距离接近 ≠ 上游身份相同。
4. **反向同样成立**：行为不一致也可能来自服务端版本升级、系统提示变化、
   采样/负载差异，而不是"换了个模型"。

因此实现层的结论等级被设计为**四级、且默认最弱**：

| 等级 | 触发条件 | 含义 |
|---|---|---|
| `INCONCLUSIVE` | 没有任何可比参考 | 没有可排序证据 |
| `RANKED_ONLY` | 有排序，但无 validated policy / 无被声称身份的参考 | **有排序、无判定** |
| `BEHAVIOR_CONSISTENT_WITH_CLAIM` | validated policy + 被声称身份参考 + 距离 ≤ threshold | 行为与声称一致 |
| `BEHAVIOR_INCONSISTENT_WITH_CLAIM` | validated policy + 被声称身份参考 + 距离 > threshold | 行为与声称不一致 |

模型层强制：claim verdict 状态**必须**携带产生它的 policy 身份与 claimed 参考距离；
`claim_verdict_produced` 与 `match_status` 不一致即构造失败。

---

## 6. 关于 mock E2E 中的 `Threshold = 0.0000`

端到端测试（`tests/integration/test_fingerprint_e2e.py`）里会出现
`Threshold: 0.0000 / TPR: 1.00 / FAR: 0.00 / Validated YES`。这是**确定性 fixture 的属性**：

- 测试使用确定性 mock（每个 identity 对每个 probe 给出固定选项），
  因此"自身 identity"的距离恰好为 `0.0`，而 impostor 距离显著大于 `0`；
- 在 `FAR <= max_far` 约束下，`0.0` 恰好是一个可行候选并被选中。

**这不表示 production threshold 应该是 0，也不表示真实部署应该追求 0 阈值。**
真实参考数据存在采样噪声与版本漂移，`llmtrace fingerprint validate` 会在**该参考集自己**的
held-out 分布上选出满足 FAR 约束的 threshold；不同参考集得到不同 threshold，属正常现象。
把 `0.0000` 当作"正确配置"会得出错误的工程结论。

---

## 7. 与外部研究的关系（attribution）

本实现为 **clean-room**：只借鉴方法论层面的公开思想，**不复制任何外部源码**，
也**不引入**其 runtime 或数据模型。采纳的方法论点（JSD 分布匹配、held-out 验证、
Top-K 参考比对、逐 probe 覆盖率可审计）已在
[`docs/research/external-influences.md`](../research/external-influences.md) 中逐项点名致谢；
该台账同时标明：对未能核验到明确 `LICENSE` 的来源，记为
**license not relied upon for implementation**，不猜测其许可类型。
本轮实现**不依赖**任何外部项目的许可证。

---

## 8. Future research（仅登记，不在本轮实现）

以下方向**本轮不实现**，仅作为后续研究登记；本轮实现不含其代码、参数或数据：

- **Rank-Based Uniformity Test**（arXiv:2506.06975）：基于 rank 均匀性的统计检验，
  用于检测"输出分布是否被人为压制/截断"。
- **LLMPrint 式更强的统计验证**（arXiv:2509.25448）：更强的统计检验与置信区间框架，
  替换当前"单阈值 + FAR 约束"的判定形式。
- **natural-query 反规避探测**：使用自然语言查询而非固定选择题，降低"选择题可被专门优化/规避"的风险。
- **open-set model verification**：不预设封闭参考集，回答"是否属于库中任何已知身份"。
- **mixture / routing 统计估计**：在统计上区分"观察到不一致"与"可以证明的路由混合"，
  并给出可校准的区间而不是伪造比例（参见 [`routing.md`](./routing.md)）。

这些方向共用同一纪律：**不抄外部研究的准确率数字**（Rule 6），任何阈值/准确率只能来自
LLMTrace 自己在参考集上的 held-out 验证。

---

## 9. 术语与措辞纪律

| 允许（行为层） | 禁止（定罪式） |
|---|---|
| behavioral identity evidence | detect fake models |
| claim consistency / behavior consistent with claim | prove fraud / 已证明作弊 |
| trusted fingerprint reference | identify the real upstream model |
| experimental model verification | 已检测到模型被偷换 |
| routing inconsistency observed / mixed routing is possible | 70% A / 30% B 混合路由 |

报告中必须与身份证据同时出现免责声明：
`Behavioral fingerprint evidence only; not cryptographic proof of upstream model identity.`
