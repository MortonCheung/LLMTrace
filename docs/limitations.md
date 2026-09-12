# LLMTrace 限制说明

## 证据链限制

- 每条证据带唯一 `evidence_id`（UUID），Finding 的 `evidence_refs` 引用此 ID，报告生成前进行完整性校验
- 响应哈希基于完整原始响应字节计算（SHA-256），摘要再按字节截断，截断不影响哈希值；`response_body_size` 为原始字节长度
- 正常基线（`baseline`）与流式对照（`streaming_comparison`、`streaming_baseline`）属于不同证据类型，互不混入
- 流式证据的 `first_token_latency_ms` 从首段非空文本 delta 计算，连接事件和空文本不计入
- 流式发生解析错误时保留已采集事件，不丢弃整条证据
- `request_id` 从多个响应头（request-id / x-request-id / anthropic-request-id / openai-request-id / x-amzn-requestid / cf-ray）中提取，`response_id` 从响应体解析，两者分开保存
- 请求次数以统一 AuditPlan 为准：dry-run 计划次数与实际执行一致；报告中的证据数与 Mock Server 调试接口返回的实际请求计数独立核验

## 客户端视角限制

- 客户端无法看到中转站真实上游后台，所有判断基于客户端可观测的协议行为
- `model` 字段和请求 ID 可以被中转站任意改写
- 中转站可以在请求到达上游之前或响应返回客户端之前修改任何字段

## 探针限制

- 无效模型被拒绝不代表实际模型真实 —— 服务商可能只做了简单的模型名白名单校验
- 无效模型成功生成内容不能区分是"忽略 model 参数"还是"默认回退到固定模型"
- 回答风格相似不等于模型身份相同 —— 不同模型可能针对简单提示给出相似回答
- 流式与非流式不一致可能来自协议实现差异，而不是模型切换

## 延迟限制

- 延迟受网络、负载、缓存和队列影响，不能作为模型身份的唯一依据
- 相同模型在不同负载下延迟差异可能很大
- 不同模型在相同网络条件下延迟可能相近

## 模型别名限制

- 模型别名和版本映射可能导致名称不一致，但并非欺诈行为
- 例如 `gpt-4` 可能映射到 `gpt-4-0613`，这是正常的版本映射
- 首版不自动区分别名和欺诈

## 风险等级限制

- LOW：本次有限测试未发现明显异常，不代表已证明真实上游模型身份
- MEDIUM：检测到中等级别的异常证据，建议进一步调查，可能包括模型列表不一致、元数据不稳定等
- HIGH：检测到高风险证据（无效模型成功返回、模型标识不一致等），接口行为存在重大异常
- INCONCLUSIVE：无法得出有效结论，可能因为鉴权失败、网络不可达或样本不足

## 统计限制

- 首版只做可解释的规则统计，不实现机器学习分类
- 稳定性分析仅针对 baseline 证据，不混入模型列表、无效模型、流式兼容性请求
- 样本量有限（默认 3 次重复），统计结论可能不具代表性
- 报告中的证据数对应真实 HTTP 请求数（每条请求一条证据），但请求计数以 AuditPlan 和 Mock Server 调试接口为准，不以报告文字为准
- 跨报告比较受测试套件版本和接口变化影响
- v0.5 及以前不具备模型指纹识别能力；v0.6 起提供**实验性行为身份证据**，
  但该证据不是身份证明，也不能回答"真实上游模型是什么"（见下节）

## Quick Suite 与校准分的可靠性语义

- **32 题 Quick Suite 是 low-cost screening measurement（低成本筛查测量）**，
  用于在不消耗大量预算的前提下给出可复现的能力坐标，不追求与全量基准同等的测量精度。
- **正式 reference calibration 使分数在已定义的 Reference Universe 内可复现**
  （同一 ReferenceSet + 同一 CalibrationPolicy + 同一 suite 下，同一测量得到同一分数），
  但**不意味着**较小的数值差异能忠实复现 full-benchmark ranking —— 32 题上的小幅差距
  不足以支撑"模型 A 强于模型 B"这类排序结论。
- **小的 capability score 差异不得解读为 model-identity evidence**：
  能力分是能力坐标，不是身份判据；分数接近 ≠ 同一个模型，分数差异 ≠ 模型被替换。
- 30 题以内的小样本筛查会受题目抽样与随机性影响；如需排序结论，应扩大测量范围并
  在多组参考宇宙下复核，而不是引用 Quick Suite 的小数点后差异。

## 指纹与身份证据限制（v0.6，实验性）

- 指纹匹配产出的是 **behavioral identity evidence**，**不是密码学证明**，
  也不构成对上游身份的证明。
- `Top-K` 只是在**已登记的受信任参考集**中最接近的若干参考；参考集是有限且人工采集的，
  因此**不能**回答开放世界（open-set）的身份判定问题。
- `Top-1` 不得被解读为 "actual upstream model"；`similarity = 1 - distance`
  不是概率、不是置信度、不是准确率。
- 只有 **validated** `FingerprintDecisionPolicy` + 被声称身份的参考 + 足够可比证据，
  才允许输出 claim consistency；否则只有排序（`RANKED_ONLY`）或无结论（`INCONCLUSIVE`）。
- 没有兼容的 `FingerprintReferenceSet` 时 Model Verification 为 `Unavailable`，
  能力审计照常继续；这属于正常降级，不是运行失败。
- 低熵选择题的行为分布可被提示包装、采样参数与后处理改变；
  行为一致/不一致都可能来自版本升级、系统提示变化或负载差异，而非模型替换。
- routing 报 `Suspicious` 是审计结论，**不等于 run 失败**，也**不构成**已证明的混合路由
  （不输出任何混合比例）。详见 [`docs/methodology/routing.md`](./methodology/routing.md)。

## 协议限制

- 首版仅支持 OpenAI-compatible 和 Anthropic-compatible 协议
- 不支持 Gemini、Cohere 等非标准协议
- 流式解析对非标准 SSE 实现可能不完整