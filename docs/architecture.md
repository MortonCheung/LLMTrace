# LLMTrace 架构说明

## 总体架构

独立 Python 核心 + 协议证据采集 + 中转站专项探针 + 风险解释 + 统一报告。

## 模块职责

### Provider 层 (`src/llmtrace/providers/`)

负责与外部 API 交互，生成符合协议的请求、发送请求、解析协议字段、返回统一证据对象。

- `base.py`：定义 Provider 抽象接口（`list_models`、`complete`、`stream_complete`）
- `openai_compatible.py`：OpenAI-compatible 协议实现
- `anthropic_compatible.py`：Anthropic-compatible 协议实现
- `url_utils.py`：安全 URL 路径拼接

### Probe 层 (`src/llmtrace/probes/`)

基于 Provider 返回的证据进行测试，每个探针输出结构化发现结果。

- 配置预检、连接与鉴权、模型列表、正常基线、无效模型、流式一致性、元数据完整性、会话稳定性

### Analysis 层 (`src/llmtrace/analysis/`)

风险分析、结构指纹、跨报告漂移比较。

- `risk.py`：基于探针结果计算风险等级
- `schema_fingerprint.py`：基于 JSON 结构生成稳定指纹
- `drift.py`：比较多份报告，检测漂移

### Reporting 层 (`src/llmtrace/reporting/`)

终端摘要、JSON 报告、HTML 报告生成。

### 数据模型 (`src/llmtrace/models/`)

Pydantic 模型定义审计配置、HTTP 证据、发现结果、报告。

## 数据流

```
用户 CLI 输入 → AuditConfig → Provider 层 → HTTP 请求 → 响应解析
→ HTTPEvidence → Probe 层 → FindingResult → Analysis 层
→ RiskLevel → Reporting 层 → JSON/HTML 报告
```

## 为什么首版不以 Promptfoo 为核心

- Promptfoo 面向提示词评估和红队测试，不是中转站审计
- 需要额外适配才能采集原始协议证据
- 无法直接实现无效模型回退检测
- 后续可接入 Promptfoo 作为外部引擎，提升评测深度

## 后续如何接入外部引擎

- v0.5：通过 LLMmap 指纹库进行模型指纹匹配
- v0.6：适配 Model Equality Testing 统计方法
- v0.7：通过 Promptfoo 和 lm-eval 扩展评测能力