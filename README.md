# LLMTrace - 模型寻迹

面向第三方 AI API、中转站和代理服务的黑盒模型审计工具。

## 当前版本能做什么

v0.1 实现首个纵向闭环：输入中转站地址、协议、模型名称和密钥环境变量，自动采集原始协议证据、运行基础探针、计算风险、输出终端摘要，并生成可复查的 JSON 与 HTML 报告。

### 核心能力

- 原始协议证据采集（请求/响应头、响应体、延迟、Token 等）
- 无效模型静默回退检测
- 元数据一致性与完整性检查
- 流式与非流式接口一致性对比
- 单次会话稳定性分析
- 多报告跨时间漂移比较
- 密钥自动脱敏，不写入报告

## 当前版本不能证明什么

- 不能识别中转站真实上游模型
- 不能证明服务商是否使用了声明模型
- 低风险结果不代表服务商已被证明可信
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