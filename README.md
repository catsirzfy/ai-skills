# AI Application Development Portfolio

AI 应用开发技术能力展示，涵盖 LLM API 集成、Prompt Engineering、MCP 工具协议、RAG 检索增强生成、Agent 构建、多 Agent 协作、企业级项目脚手架。

## 技术能力

| 模块 | 核心技术 | 产出 |
|------|---------|------|
| LLM 应用 | OpenAI SDK、DeepSeek、流式输出、Token 管理 | API 客户端 + 对话引擎 + 提示词模式库 |
| MCP 协议 | 工具定义、JSON-RPC、Function Calling | MCP Server/Client + 4 工具完整流程 |
| RAG 引擎 | LangChain/LCEL、ChromaDB、混合检索、重排序、多轮对话 | 手写引擎 + LlamaIndex 版 + LangChain 版 |
| Agent | LangChain Agent、LangGraph 状态图、ReAct 循环 | 多工具 Agent + 数据分析 Agent |
| 多 Agent | 串行/并行协作、角色分工 | 代码审查团队 + 研究员-写作者模式 |
| 工程化 | FastAPI + SSE + Docker | 企业级脚手架 + 完整项目模板 |

## 项目结构

```
├── 01-llm-applications/     # LLM API 集成 & 提示词工程
│   ├── api_client.py        #   多平台适配（DeepSeek/通义/Ollama）
│   ├── prompt_patterns.py   #   4 种提示词模式库
│   └── interactive_chat.py  #   流式对话引擎（会话持久化）
├── 03-mcp/                  # MCP 工具协议
│   ├── mcp_server.py        #   MCP Server（4 个工具）
│   └── mcp_client.py        #   AI + Tool Calling 完整流程
├── 04-rag/                  # RAG 检索增强生成
│   ├── rag_engine.py        #   完整 RAG 管道 + 混合检索 + 重排序 + 多轮对话
│   ├── llamaindex_demo.py   #   LlamaIndex 框架实现
│   ├── langchain_demo.py    #   LangChain 框架实现（两种方式）
│   ├── advanced_demo.py     #   高级特性演示
│   └── api_server.py        #   FastAPI + SSE 流式 API
├── 05-agent-skills/         # Agent 核心能力
│   ├── tool_use.py          #   Function Calling 工具调用
│   ├── memory.py            #   Buffer/Window/Summary 三种记忆
│   ├── planning.py          #   ReAct / Plan-Execute / 任务分解
│   └── streaming.py         #   流式输出 / SSE 格式
├── 06-agent/                # 完整 Agent 应用
│   ├── langchain_agent.py   #   LangChain AgentExecutor
│   ├── langgraph_agent.py   #   LangGraph 状态图 Agent
│   └── data_agent.py        #   数据分析 Agent（求职作品）
├── 07-multi-agent/          # 多 Agent 协作
│   ├── multi_agent_langgraph.py  # 串行+并行 Agent
│   └── multi_agent_crew.py       # CrewAI 风格代码审查团队
└── 08-projects/             # 实战项目
    ├── fastapi-scaffold/    #   企业级 FastAPI 脚手架
    └── my-first-app/        #   脚手架实战（全栈 AI 对话应用）
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python 3.12+ |
| LLM SDK | OpenAI SDK（兼容 DeepSeek / 通义千问 / Ollama） |
| AI 框架 | LangChain / LangGraph / LlamaIndex |
| 向量数据库 | ChromaDB |
| Embedding | BAAI/bge-small-zh（本地），text-embedding-3-small（API） |
| Web 框架 | FastAPI + SSE 流式输出 |
| 部署 | Docker + Docker Compose |

## 快速开始

```bash
pip install openai python-dotenv
cp .env.example .env          # 填入 DEEPSEEK_API_KEY
python 01-llm-applications/api_client.py
```

## License

MIT
