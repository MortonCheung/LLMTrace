# Real Reference Validation — v0.6-R1 研究轮（Real Fingerprint Campaign 设计与状态）

本文件定义并记录 v0.6-R1 研究轮：用**真实可信模型**建立第一批真实 Fingerprint Reference
Data，检验 `llmtrace-fingerprint-categorical 0.1.0` 在真实世界是否具有足够的稳定性和区分度。

它回答的问题与 `docs/methodology/fingerprinting.md`（实现方法）不同：本文档是**实验设计
与实验状态记录**——包括数据如何构造、指标如何定义、以及结果如何解释。

> **本轮性质**：这不是正式功能 milestone。目标是"证明方法真的有用"而不是"继续开发新模块"。
> 即使结论是「Fingerprint v1 当前真实区分能力不足」，也是合法且有价值的结果。

---

## 1. Experiment Question（实验问题）

核心问题：

```text
同一模型不同 capture 之间的距离（within），
与不同模型之间的距离（between），
是否被当前 probe suite 明显分开？
```

由此派生：

- 哪些 probe 真正具有区分力，哪些基本没有信息，哪些不稳定？
- INVALID 输出占多少？
- STANDARD=8（每 probe 8 次采样）是否足够？
- 真实 held-out Top-1 / FAR / TPR 是多少？
- Fingerprint v1 当前能否安全支撑 claim consistency 判定？

## 2. Dataset Construction（数据集构造）

最低实验集（本轮 blocker 定义）：

```text
5 个 distinct identities × 2 次独立 capture = 10 captures
```

STANDARD profile 下每 capture 请求数为 `len(suite.probes) × repetitions = 6 × 8 = 48`
（程序内一律由该公式计算，不硬编码 48）。总计约 480 次真实请求。

推荐扩展（非 blocker）：5 identities × 3 captures，第三次用于跨 session 稳定性观察。

### 2.1 冻结的实验条件

真实 capture 开始前记录并冻结（本轮记录）：

| 条件 | 值 |
|---|---|
| suite_id | `llmtrace-fingerprint-categorical` |
| suite_version | `0.1.0` |
| suite_content_sha256 | `a34f551be0cc4451ea3b9c4abd10bebb345991622d3f90210a9500b81bd8ab3b` |
| normalization_policy | `llmtrace-exact-choice` `1.0.0` |
| generation_config_sha256 | `46c3b412431504dd456f31c581be93c6c98d65991c0776ec4d2f5e5f274e7aa7` |
| profile | `standard`（repetitions = 8） |
| probe_count | 6 |

### 2.2 Capture 纪律

- 两次 capture 必须是**两个独立 CLI execution**，不得复制同一 snapshot。
- 每次 capture 后立即用 `llmtrace fingerprint inspect` / repository API 执行完整性门禁
  （self hash、suite hash、generation config、profile、repetitions、probe count、sample
  count、source_role、provider_id、model_id）。
- 失败 capture（429 / timeout / 大量 invalid / HTTP failure）：**不手工改 JSON**。保留原
  snapshot，需要重采时使用新的 snapshot_id（如 `fp-model-a-capture-02-retry01`），不覆盖历史。

## 3. Identity Definition（身份定义）

验证身份继续使用 `(provider_id, model_id)` 二元组，**不是**单独的 `model_id`：

```text
provider_id = official-openai, model_id = model-a
provider_id = relay-x,         model_id = model-a
```

是两个不同 Reference identity。分析工具 `tools/analyze_fingerprint_references.py` 的
`identity_key()` 实现即此语义。

## 4. Capture Profile（成本档位）

| profile | repetitions | 每 capture 请求（6 probes） |
|---|---|---|
| quick | 4 | 24 |
| standard | 8 | 48 |
| research | 16 | 96 |

profile 只表示成本，不表示准确率。本轮主实验统一 `standard`；若预算允许，另选至少 2 个
identities 在 quick / research 下**独立**建 set 对比（Task 25，可 defer），不与主实验
混入同一 ReferenceSet。

## 5. Reference Provenance（参考来源）

| role | 条件 | 示例 |
|---|---|---|
| `official-baseline` | Owner 明确确认：endpoint 属于模型厂商官方服务、key 来自官方账户、model id 是官方当前真实 label | 官方 API 直连 |
| `trusted-reference` | 可信但非厂商官方 endpoint | 可信自建网关 |
| （禁止） | 第三方 relay / 中转站 / 未知代理 | 不得作为 official-baseline |

真实密钥只允许存在于环境变量（如 `OPENAI_REFERENCE_KEY`）；实验文件与命令行只记录
`api_key_env` 变量名，绝不记录值。campaign manifest 模板见
`docs/examples/fingerprint-campaign.example.json`；真实 manifest 建议放
`.local/`（已被 `.gitignore` 忽略），不入 Git。

## 6. Metrics（指标定义）

全部由分析工具从 snapshot repository **只读**计算（复用生产
`compute_fingerprint_distance`，不重新实现 JSD）：

### 6.1 Within / Between 分析

- **within**：同 `(provider_id, model_id)` 两次 capture 之间的加权 JSD 距离。
- **between**：不同 identity capture 之间的距离。
- **conservative separation margin** = `min(between) - max(within)`：最接近的一对不同模型，
  仍然比最不稳定的一对同模型更远。`margin > 0` 是好现象，但**不等于**已证明泛化。

### 6.2 Probe Discrimination（逐 probe 区分度）

- `within-probe JSD` / `between-probe JSD`：单 probe 独立统计。
- **probe separation** = `mean_between_jsd - mean_within_jsd`（第一版研究指标）。
  它只是 **research discrimination metric**，不得称为 accuracy 或 confidence。
- 研究标签（阈值：`max_invalid_rate = 0.20`、`max_within_mean = 0.15`、
  `min_separation = 0.05`）：
  `PROMISING` / `WEAK` / `UNSTABLE` / `INVALID_HEAVY` / `INSUFFICIENT_DATA`。

> These thresholds are exploratory labels only and are not production identity
> decision thresholds. 研究标签不进入 `FingerprintDecisionPolicy` 或任何生产判定。

### 6.3 INVALID 分析

每个 `identity × probe` 计算 `invalid_count / sample_count`，再聚合全局 probe invalid rate。
INVALID 不删除——经常 INVALID 的 probe 可能仍有 fingerprint 信息，但必须单独报告。

### 6.4 Held-out Validation

由 `llmtrace fingerprint validate` 执行 leave-one-capture-out 验证，产出
`FingerprintDecisionPolicy`（只有 `validated = YES` 的 policy 才能用于 claim consistency）。
记录：identity_count、capture_count、Top-1 / Top-3、TPR / FAR、threshold、全部 held-out
own_distance 与 nearest_impostor_distance。

**不得为了结果漂亮调整 threshold**（不调 `--max-far`，优先用正式默认值 0.01）。

## 7. Data Leakage Policy（数据泄漏政策）

- 若本轮数据被用于挑 probe / 调 temperature / 改 choices / 调 weights，这些数据从此属于
  **development set**；后续 suite 0.2.0 必须使用**新的独立 captures** 重新验证。
- 禁止"用同一批数据挑 probe，然后仍用同一批数据宣称验证成功"。
- probe-development data ≠ held-out final validation data。本轮明确：
  `fingerprint_v1.json` 在 Reference Campaign 开始后完全冻结；发现问题只记录，改动留给
  0.2.0 并需新数据。

## 8. Decision Gate（决策门）

最终只允许三种结论（详见任务记录 Task 23）：

| Outcome | 判据（摘要） | 后续 |
|---|---|---|
| **KEEP** | held-out validation 成功、FAR 满足 policy、Top-1 稳定、same/between 有明显 separation、无明显单个 identity 崩坏 | 进入真实 relay validation |
| **TUNE** | 整体有区分力但 2–3 个 probe 弱或 STANDARD=8 方差大 | 规划 suite 0.2.0（新独立数据） |
| **REDESIGN** | within/between 大幅重叠、Top-1 接近随机、FAR 无法满足、多模型分布几乎一样 | 研究 natural-query probes / LLMPrint-style prompts / RUT / logprob fingerprints 等 |

不能通过降低 FAR 要求、删除异常 capture、删除 INVALID 或挑最好看的模型让结果"通过"。

## 9. Real Relay Smoke Test（条件任务）

只有真实 fingerprint policy 通过 validation 后才执行：选 1–3 个真实第三方 endpoint 运行
`llmtrace run --verify-model`，记录 Top-K、claimed reference distance、claim verdict、
routing v2、response_model、invalid count、protocol findings、capability gap。

即使出现 `BEHAVIOR_INCONSISTENT_WITH_CLAIM`，报告也只能写：

> The observed behavior is inconsistent with the trusted claimed-model reference under the
> validated fingerprint policy.

禁止"该中转站是假模型 / 该供应商欺诈 / 真实模型一定是 X"的表述。Top-1 也只能叫
**closest behavioral reference**。

## 10. 本轮实际状态（2026-09-13 记录）

按 Task 31 如实记录：

```text
REAL DATA CAMPAIGN: PARTIAL

Available identities: 0（当前环境无 Owner 提供的真实 API 凭据）
Required: >= 5

No production fingerprint validation was claimed.
```

- 已完成：campaign 设计文档（本文件）、campaign manifest 模板、只读分析工具
  `tools/analyze_fingerprint_references.py`（Task 14-22）及其测试（Task 29）。
- 未执行：真实 capture（Task 9-13）、真实 held-out validation、relay smoke test——
  均因无真实凭据而 defer，**Mock 未被用于补齐真实 identity 数量**。
- 真实参考数据默认留本地（`~/.llmtrace/fingerprints/`），不入 Git；本轮 Git 只包含
  analysis tooling、example manifest、methodology 与 anonymized/approved 聚合报告。

## 11. Limitations（局限）

- 即使 10 个 captures 全部完成，每 identity 仅贡献 1 个 within pair——within 方差的估计
  统计上极弱；长期稳定性需跨 session / 跨时间重新采集。
- `margin > 0` 是样本级分离，不是泛化证明；真实世界 routing / 采样噪声 / 服务端变化
  都可能侵蚀 margin。
- 分类 probe 的区分力天花板：如果多个真实模型在这些低熵分类上分布几乎一致，v1 的
  categorical 方法本身（而非实现）就不足——那属于 REDESIGN 范畴，不是 bug。
- Fingerprint 行为证据不是密码学证明（同 `docs/methodology/fingerprinting.md` 的边界声明）。
