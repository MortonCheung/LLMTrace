# LLMTrace - 模型寻迹

面向第三方 AI API、中转站和代理服务的黑盒模型审计工具。

## 当前版本能做什么

v0.1 实现首个纵向闭环：输入中转站地址、协议、模型名称和密钥环境变量，自动采集原始协议证据、运行基础探针、计算风险、输出终端摘要，并生成可复查的 JSON 与 HTML 报告。

### 核心能力

- 统一执行链：每次请求只发送一次，探针直接返回真实证据（Evidence）与发现（Finding），CLI 仅负责编排汇总
- 每条证据带唯一 `evidence_id`（UUID），所有 Finding 的 `evidence_refs` 引用真实存在的证据 ID
- 证据按类型分类（baseline / invalid_model / streaming_baseline / model_catalog / connectivity），稳定性与元数据分析仅针对基线证据
- 无效模型探针使用随机假模型名请求，证据中同时记录配置声明模型、本次随机无效模型和服务端返回模型
- 响应哈希基于完整响应体计算后再截断保存摘要，截断时标记 `truncated=true`
- 流式首 Token 延迟从首段非空文本 delta 计算，记录流开始/首文本/结束时间
- 多报告跨时间漂移比较
- 密钥自动脱敏，不写入报告

## 当前版本不能证明什么

- 不能识别中转站真实上游模型（当前版本不具备模型指纹识别能力）
- 不能证明服务商是否使用了声明模型
- LOW 风险等级只表示"本次有限测试未发现明显异常，不代表已证明真实上游模型身份"
- 单次延迟不能直接判断模型身份

## 安装

```bash
git clone https://github.com/MortonCheung/LLMTrace.git
cd LLMTrace
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

要求 Python 3.11+。

## 使用示例

### OpenAI-compatible 接口审计

```bash
export OPENAI_API_KEY="your-key"
llmtrace audit \
  --protocol openai \
  --base-url https://api.example.com \
  --model gpt-4 \
  --api-key-env OPENAI_API_KEY
```

### Anthropic-compatible 接口审计

```bash
export ANTHROPIC_API_KEY="your-key"
llmtrace audit \
  --protocol anthropic \
  --base-url https://api.example.com \
  --model claude-3-opus-20240229 \
  --api-key-env ANTHROPIC_API_KEY \
  --auth-style x-api-key
```

### Dry-run（不发送请求，仅显示执行计划）

```bash
llmtrace audit --protocol openai --base-url https://api.example.com --model test --api-key-env OPENAI_API_KEY --dry-run
```

### 查看报告

```bash
llmtrace inspect reports/llmtrace_20260804_120000_abc123.json
```

### 比较报告

```bash
llmtrace compare reports/run_a.json reports/run_b.json
```

## 模拟服务器

项目包含一个本地模拟服务器，用于不消耗 API 的端到端验证：

```bash
python examples/mock_proxy_server.py --mode honest --port 8080
```

支持三种模式：
- `honest`：正常模型成功，无效模型返回 404
- `fallback`：不论请求什么模型都成功，暴露静默回退行为
- `inconsistent`：轮换返回模型、usage 字段时有时无

## 报告示例

每次审计生成同名 JSON 和 HTML 报告。JSON 报告包含完整脱敏证据，HTML 报告可本地打开，不依赖 CDN。

## 密钥安全

- 密钥仅允许来自环境变量或隐藏交互式输入
- 禁止命令行明文 `--api-key`
- 报告自动脱敏所有鉴权信息
- 不写入日志或异常堆栈

## 当前路线图

| 版本 | 内容 |
|------|------|
| v0.1 | 证据审计 MVP（当前版本） |
| v0.2 | 轻量能力基准 |
| v0.3 | 官方参考模型对照 |
| v0.4 | 跨时间路由漂移 |
| v0.5 | LLMmap 指纹适配 |
| v0.6 | Model Equality Testing 统计模式 |
| v0.7 | Promptfoo 与 lm-eval 外部适配 |
| v0.8 | Web 可视化界面 |