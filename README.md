# AI Application Development Portfolio

AI 应用开发技术能力展示，覆盖 LLM API 集成、Prompt Engineering、RAG、Agent 构建等核心技能。

## 技术能力概览

| 能力领域 | 核心技术 | 对应模块 |
|---------|---------|---------|
| **LLM 应用开发** | OpenAI SDK / DeepSeek / 流式输出 / Token 管理 | `01-llm-applications` |
| **本地模型部署** | Ollama / 本地 API 服务 | (进行中) |
| **MCP 工具集成** | MCP Server/Client / 外部工具调用 | (进行中) |
| **RAG 知识库** | 向量数据库 / 文档解析 / 混合检索 | (进行中) |
| **Agent 构建** | Function Calling / ReAct / LangGraph | (进行中) |

## 项目结构

```
├── 01-llm-applications/        # LLM API 集成 & Prompt Engineering
│   ├── api_client.py           #   多平台 API 客户端封装
│   ├── prompt_patterns.py      #   结构化提示词模式库
│   └── interactive_chat.py     #   流式对话引擎（支持会话持久化）
├── 02-ollama/                  # 本地模型部署
├── 03-mcp/                     # MCP 协议
├── 04-rag/                     # RAG 检索增强生成
├── 05-agent-skills/            # Agent 核心能力
├── 06-agent/                   # 完整 Agent 应用
├── 07-multi-agent/             # 多 Agent 协作
└── 08-projects/                # 综合实战项目
```

## 01 — LLM API 集成 & Prompt Engineering

### 能力要点

- **多平台 API 适配**：统一的 OpenAI 兼容接口，无缝切换 DeepSeek / 通义千问 / Ollama
- **流式输出**：基于 SSE 的实时流式响应
- **会话管理**：完整的对话上下文维护、持久化与恢复
- **结构化输出控制**：Temperature 精确调控 + JSON 格式化约束
- **提示词模式**：角色扮演 / Chain-of-Thought / 结构化输出 / 多轮记忆 四大模板体系

### 运行

```bash
pip install openai python-dotenv

# 配置 API Key（推荐 DeepSeek，国内直连）
cp .env.example .env  # 编辑 .env 填入 key

python 01-llm-applications/api_client.py
python 01-llm-applications/prompt_patterns.py
python 01-llm-applications/interactive_chat.py
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python 3.12+ |
| LLM SDK | OpenAI SDK（兼容 DeepSeek / 通义千问 / Ollama） |
| 框架（计划） | LangChain / LangGraph / LlamaIndex / FastAPI |
| 向量数据库（计划） | Chroma / Milvus |
| 部署（计划） | Docker |

## License

MIT
