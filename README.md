# AI Application Development Portfolio

AI 应用开发技术能力展示，涵盖 LLM API 集成、Prompt Engineering、MCP 工具协议、RAG 检索增强生成、Agent 构建。

## 学习路线

```
Skills 基础 → Ollama 本地部署 → MCP 工具协议 → RAG 知识库 ★ → Agent Skills → Agent 完整体 ★ → 多Agent → 实战项目
```

---

## 项目结构

```
├── 01-llm-applications/        # LLM API 集成 & 提示词工程
│   ├── api_client.py           #   多平台适配（DeepSeek/通义/Ollama）
│   ├── prompt_patterns.py      #   4 种提示词模式库
│   └── interactive_chat.py     #   流式对话引擎（会话持久化）
├── 02-ollama/                  # 本地模型部署（待完成）
├── 03-mcp/                     # MCP 工具协议
│   ├── mcp_server.py           #   MCP Server（4 个工具）
│   └── mcp_client.py           #   AI + Tool Calling 完整流程
├── 04-rag/                     # RAG 检索增强生成 ★核心
│   ├── rag_engine.py           #   完整 RAG 管道（载入→分块→向量→搜索→生成）
│   └── demo.py                 #   Demo：索引本地文档并问答
├── 05-agent-skills/            # Agent 核心能力（待完成）
├── 06-agent/                   # 完整 Agent 应用（待完成）
├── 07-multi-agent/             # 多 Agent 协作（待完成）
└── 08-projects/                # 综合实战项目（待完成）
```

---

## 快速开始

```bash
# 1. 安装依赖
pip install openai python-dotenv numpy

# 2. 配置 API Key（推荐 DeepSeek，国内直连）
cp .env.example .env
# 编辑 .env，填入你的 DEEPSEEK_API_KEY
# 注册获取 key: https://platform.deepseek.com

# 3. 运行各模块
python 01-llm-applications/api_client.py          # API 调用入门
python 01-llm-applications/prompt_patterns.py      # 提示词模式演示
python 01-llm-applications/interactive_chat.py     # 交互式对话引擎
python 03-mcp/mcp_client.py                        # MCP 工具调用演示
python 04-rag/demo.py                              # RAG 知识库问答演示
```

---

## 各模块详解

### 01 — LLM API 集成 & 提示词工程

**学到的技能**：OpenAI 兼容协议、流式输出、Token 管理、提示词设计

| 文件 | 演示内容 | 核心知识点 |
|------|---------|-----------|
| `api_client.py` | 基础调用 / system prompt / 流式输出 | messages 三段式、temperature 控制 |
| `prompt_patterns.py` | 角色扮演 / 链式思考 / JSON 输出 / 多轮记忆 | 4 种可复用提示词模板 |
| `interactive_chat.py` | 命令行聊天机器人 | 对话状态管理、会话持久化 |

**关键概念**：
- **Messages**：`system`（人设）+ `user`（用户）+ `assistant`（AI）三段式是 API 调用的核心
- **Streaming**：`stream=True` 逐字返回，用户体验远好于一次性等待
- **Temperature**：0 = 稳定重复，0.7 = 平衡，1.5 = 创意

### 03 — MCP 工具协议

**学到的技能**：MCP 协议原理、Function Calling、Tool Use 循环

| 文件 | 演示内容 | 核心知识点 |
|------|---------|-----------|
| `mcp_server.py` | MCP Server 实现（4 个工具） | JSON-RPC、工具注册与路由 |
| `mcp_client.py` | AI 自动选择并调用工具 | Function Calling、两轮调用模式 |

**关键概念**：
- **MCP（Model Context Protocol）**：让 AI 通过标准协议调用外部工具的开放标准
- **Tool Use 循环**：AI 决定调工具 → 执行 → 结果回传 → AI 生成最终回答
- 这正是 Agent 的核心机制——后面学 Agent 时底层原理完全相同

### 04 — RAG 检索增强生成 ★

**学到的技能**：文档处理、文本分块、向量 Embedding、语义搜索、知识库问答

| 文件 | 演示内容 | 核心知识点 |
|------|---------|-----------|
| `rag_engine.py` | 完整 RAG 管道实现 | 分块策略、余弦相似度、向量持久化 |
| `demo.py` | 索引本地文档并问答 | 文档索引 → 语义搜索 → AI 生成 |

**关键概念**：
- **完整流程**：文档 → 分块 → Embedding → 向量库 → 搜索 → AI 回答
- **Embedding**：文本 → 1536 维向量，语义相近的文本向量也相近
- **余弦相似度**：衡量两个向量"有多像"的数学公式
- **分块策略**：500 字一块 + 80 字重叠，平衡语义完整性和检索精度

---

## 核心概念速查

| 概念 | 一句话 |
|------|--------|
| Messages | `system`(人设) + `user`(你) + `assistant`(AI)，对话的核心数据结构 |
| Temperature | 0=严谨, 0.7=平衡, 1.5=创意。代码用低温，创意用高温 |
| Streaming | `stream=True`，AI 像打字机一样逐字输出 |
| System Prompt | 设定 AI 行为的最有效方式，角色扮演的核心 |
| Function Calling | AI 决定调用哪个函数，传什么参数——Agent 的基础 |
| MCP | 连接 AI 和外部工具的标准化协议 |
| RAG | 检索相关文档 → 喂给 AI → AI 基于文档回答 |
| Embedding | 把文字变成数字向量，语义相近的向量也相近 |
| Chunking | 把长文档切成小块，每块 500 字左右 |
| Agent | AI 自主决定用什么工具、执行什么步骤 |

---

## API 选择

| 服务 | 访问 | 价格 | 用途 |
|------|------|------|------|
| DeepSeek | 国内直连 | 便宜 | 日常开发首选 |
| 通义千问 | 国内直连 | 便宜 | 阿里云用户 |
| Ollama | 本地免费 | 免费 | 开发调试、离线场景 |
| OpenAI | 需代理 | 较贵 | 最强能力 |

---

## 技术栈

| 层级 | 当前 | 后续加入 |
|------|------|---------|
| LLM SDK | OpenAI SDK | — |
| 向量计算 | numpy | Chroma / Milvus |
| 框架 | — | LangChain / LangGraph / LlamaIndex |
| Web 服务 | — | FastAPI + SSE 流式 |
| 部署 | — | Docker |

## License

MIT
