# Capability Benchmark Plan

> 状态：设计阶段  
> 版本：0.1.0  
> 原则：只做正式能力题集选择设计，不接真实题库。  
> 约束：不下载数据集、不调用真实 API、不在此 PR 内接入正式题库。

---

## 一、总体设计原则

### 1.1 评分闭环

```
Benchmark Execution → Normalized Score (0.0–1.0)
    → TaskScoringRegistry (task_id → dimension)
    → Dimension Aggregation (weighted)
    → CapabilityProfile
    → Report
```

### 1.2 Benchmark 选择标准

- **开源可审计**：数据和评分代码必须公开可复现
- **许可证兼容**：允许用于模型评估（不要求模型合规，只做黑盒 API 调用）
- **客观评分**：自动评分，不依赖人工评审
- **无 contamination 风险**：不依赖训练集泄露的固定选择题
- **适合第三方黑盒**：只需要文本输入/输出，不需要访问模型内部
- **低成本**：优先选取规模适中的评测集

---

## 二、各维度详细分析

### 2.1 Reasoning（推理能力）

**需要测什么能力：**

- 逻辑推理（归纳、演绎、溯因）
- 多步推理（需要中间步骤的链式推理）
- 因果推理
- 常识推理
- 抽象推理与类比

**题型：**

- 多选问答（需提供推理链）
- 开放式推理（自由文本，按参考答案评分）
- 基于场景的推理任务

**推荐开源 Benchmark 候选：**

| Benchmark | 推荐度 | 规模 | License | 下载 | 评分方法 | 成本 | Contamination 风险 | 黑盒适配 |
|-----------|--------|------|---------|------|----------|------|-------------------|----------|
| **ARC-Challenge** | 高 | 1,172 道 | CC BY-SA 4.0 | 是（HuggingFace） | 准确率 | 极低 | 中等 | 优秀 |
| **MMLU (STEM 子集)** | 中 | ~3,000 道 | MIT | 是（HuggingFace） | 准确率 | 低 | 高（老模型普遍见过） | 良好 |
| **BIG-Bench Hard** | 高 | ~6,500 道 | Apache 2.0 | 是（GitHub/HF） | CoT 评分 | 中 | 中等 | 良好 |
| **GSM8K** | 高 | 1,319 道 | MIT | 是（HF） | 最终答案匹配 | 低 | 中高 | 良好 |
| **LogiQA** | 中 | 651 道 | CC BY 4.0 | 是（HF/GitHub） | 准确率 | 极低 | 低 | 优秀 |

**推荐首选组合：** ARC-Challenge + GSM8K + BIG-Bench Hard（选逻辑推理子集）

**预估总规模：** ~4,000 道题

**注意：**
- MMLU contamination 风险高，适合作为辅助指标而非主力
- 部分 Benchmark 需要 CoT（Chain-of-Thought）prompt，必须使用统一 prompt 模板

---

### 2.2 Coding（编程能力）

**需要测什么能力：**

- 代码生成（从自然语言描述到正确代码）
- 代码理解（给定代码，回答关于行为的问题）
- 代码修复（给定 buggy 代码，修复）
- 算法实现（经典算法与数据结构）

**题型：**

- 函数补全（Function Completion）
- 自然语言 → 代码（Instruction → Code）
- Bug 修复（给定 failing test，补全实现）
- 代码推理（给定代码片段，预测输出）

**推荐开源 Benchmark 候选：**

| Benchmark | 推荐度 | 规模 | License | 下载 | 评分方法 | 成本 | Contamination 风险 | 黑盒适配 |
|-----------|--------|------|---------|------|----------|------|-------------------|----------|
| **HumanEval** | 高 | 164 道 | MIT | 是（GitHub） | pass@k | 低 | 高（老模型普遍见过） | 优秀 |
| **MBPP** | 高 | ~1,000 道 | CC BY 4.0 | 是（HF） | pass@k | 中 | 中高 | 优秀 |
| **LiveCodeBench** | 高 | 480+ 道 | 待确认 | 是（GitHub） | pass@k | 高（需执行） | 低（时效性强） | 优秀 |
| **BigCodeBench** | 高 | 1,140 道 | Apache 2.0 | 是（HF/GitHub） | pass@k | 高（需执行） | 中 | 优秀 |
| **APPS** | 中 | 10,000 道 | MIT | 是（GitHub） | pass@k | 高 | 中等 | 良好 |
| **DS-1000** | 中 | 1,000 道 | MIT | 是（GitHub） | pass@k | 高（需 Python 依赖） | 中 | 良好 |

**推荐首选组合：** HumanEval + MBPP（基础编程）+ LiveCodeBench（时效性保障）

**预估总规模：** ~1,500 道题

**注意：**
- pass@k 需要多次采样，估算成本时注意 k 值选择
- 代码执行沙箱安全是第一优先级
- LiveCodeBench 需定期更新（时效性强）
- HumanEval 数据可能被广泛训练过，contamination 风险高

---

### 2.3 Math & Science（数学与科学）

**需要测什么能力：**

- 算术与代数
- 几何与图形推理
- 微积分与高等数学
- 物理、化学、生物基本科学推理
- 公式推导与符号计算
- 多步数学推导与验证

**题型：**

- 数学文字题（应用题，最终答案匹配）
- 公式推导与证明
- 科学问答（多选 + 自由回答）
- 数据集理解（图表/数据解读）

**推荐开源 Benchmark 候选：**

| Benchmark | 推荐度 | 规模 | License | 下载 | 评分方法 | 成本 | Contamination 风险 | 黑盒适配 |
|-----------|--------|------|---------|------|----------|------|-------------------|----------|
| **MATH** | 高 | 12,500 道 | MIT | 是（GitHub） | 最终答案精确匹配 | 中 | 中高 | 良好 |
| **GSM8K** | 高 | 8,500 道 | MIT | 是（HF） | 最终答案匹配 | 中 | 中高 | 良好 |
| **GPQA** | 高 | 448 道 | CC BY-NC 4.0 | 是（GitHub） | 准确率 | 极低 | 低（Google 内部） | 优秀 |
| **SciBench** | 中 | 695 道 | MIT | 是（GitHub） | 工具增强评分 | 中 | 低 | 良好 |
| **MMLU-STEM** | 中 | ~3,000 道 | MIT | 是（HF） | 准确率 | 低 | 高 | 良好 |
| **TheoremQA** | 中 | 800 道 | MIT | 是（GitHub） | 最终答案匹配 | 中 | 中 | 良好 |

**推荐首选组合：** MATH + GPQA + GSM8K

**预估总规模：** ~5,000 - 8,000 道题

**注意：**
- MATH Benchmark 需要 robust 的答案解析（LaTeX 格式）
- GSM8K 既可用于 reasoning 也可用于 math_science，需在 Registry 中明确归属
- GPQA 是 Google 内部人员出题，contamination 风险最低
- CC BY-NC 4.0 需确认商业使用合规性

---

### 2.4 Instruction Following（指令遵循）

**需要测什么能力：**

- 指令精确度（是否严格按格式、顺序、约束输出）
- 多约束同时满足
- 长指令理解与执行
- 否定约束处理
- 格式约束遵循（JSON、Markdown、代码块等）

**题型：**

- 格式控制任务（如"只输出 JSON"）
- 多步骤指令（如"先翻译再总结最后分类"）
- 否定约束（如"不要使用以下词语"）
- 角色扮演与风格控制
- 结构化输出验证

**推荐开源 Benchmark 候选：**

| Benchmark | 推荐度 | 规模 | License | 下载 | 评分方法 | 成本 | Contamination 风险 | 黑盒适配 |
|-----------|--------|------|---------|------|----------|------|-------------------|----------|
| **IFEval** | 高 | 541 道 | Apache 2.0 | 是（GitHub） | 约束满足率 | 低 | 极低 | 优秀 |
| **MT-Bench** | 高 | 80 道 | Apache 2.0 | 是（GitHub） | GPT-4 Judge | 中（需 Judge 模型） | 低 | 优秀 |
| **FollowBench** | 高 | 820 道 | MIT | 是（GitHub） | 约束满足率 | 低 | 极低 | 优秀 |
| **AlpacaEval 2.0** | 中 | 805 道 | Apache 2.0 | 是（GitHub） | GPT-4 对比评判 | 高（需 Judge） | 中 | 良好 |
| **FLASK** | 中 |~1,800 道 | 待确认 | 是（GitHub） | 混合评分 | 中 | 中 | 良好 |

**推荐首选组合：** IFEval + FollowBench

**预估总规模：** ~1,500 道题

**注意：**
- IFEval 的约束检查是纯规则性的，不依赖 LLM，成本低且客观
- FollowBench 覆盖多约束场景，互补性强
- MT-Bench / AlpacaEval 需要额外的 Judge 模型，增加成本和复杂度
- 指令遵循评分的关键是"约束原子化检查"——每个约束独立验证，取满足率

---

## 三、通用讨论

### 3.1 评分方法总结

| 评分类型 | 适用场景 | 优点 | 缺点 |
|----------|----------|------|------|
| 精确匹配 | 数学题、代码输出 | 客观、快速 | 对格式敏感 |
| pass@k | 代码生成 | 统计稳定 | 需要多次采样 |
| 约束满足率 | 指令遵循 | 原子化检查 | 需要精细的约束定义 |
| 参考答案匹配 | 推理题 | 灵活性好 | 对 parser 要求高 |
| 执行验证 | 代码功能 | 最准确 | 需要安全沙箱 |

### 3.2 黑盒适配性

LLMTrace 的核心场景是第三方 API 黑盒评估。所有 Benchmark 必须满足：
- 输入仅为文本（prompt）
- 输出仅为文本（不需要概率分布、logprobs）
- 不需要模型内部权重

当前推荐的所有 Benchmark 均满足黑盒要求。

### 3.3 License 风险

- **安全 (MIT / Apache 2.0 / CC BY 4.0)**：HumanEval, MBPP, ARC-Challenge, MATH, GSM8K, BBH, IFEval, FollowBench
- **需确认 (CC BY-NC 4.0)**：GPQA（非商用条款）
- **需审查**：LiveCodeBench（需确认具体证书）

### 3.4 总规模估算

| 维度 | 推荐题数 | 预估 token 消耗 |
|------|----------|----------------|
| reasoning | ~4,000 | 中等 |
| coding | ~1,500 | 中高（含多次采样） |
| math_science | ~5,000-8,000 | 中等 |
| instruction_following | ~1,500 | 低 |
| **合计** | **~12,000-15,000** | — |

---

## 四、当前阶段不做什么

截至 Capability Scoring Foundation (v0.3)，以下事项明确不在此阶段：

- 不下载任何 Benchmark 数据集
- 不执行大规模评测
- 不调用真实外部 API
- 不建立 Reference Model 数据库
- 不输出正式 0–100 能力分
- 不构建 Web 前端
- 不添加 LLM-as-a-Judge 评分路径

以上内容将在后续阶段（v0.4+）逐步落地。
