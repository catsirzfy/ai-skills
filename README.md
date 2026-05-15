# AI 应用开发学习项目

从零开始系统学习 AI 应用开发，基于 [Python AI 应用开发学习路线](https://github.com) 构建的完整学习仓库。

## 学习路线

```
1. Skills（基础技能）           1-2 天     ★
   ↓
2. 本地模型部署（Ollama）        1 天       ★
   ↓
3. MCP（连接外部工具）           2-3 天     ★★
   ↓
4. RAG（知识库问答）★核心        2-3 周     ★★★
   ↓
5. Agent Skills（智能体核心能力） 1-2 周     ★★★
   ↓
6. Agent（构建完整智能体）★核心   2-3 周     ★★★
   ↓
7. 多 Agent 协作（了解即可）      3-5 天     ★★★★
   ↓
8. 实战项目（求职作品）           2-3 周     ★★★★
```

**核心路线约 2-3 个月，目标是做出 2-3 个能展示的项目。**

## 项目结构

```
ai-learning/
├── .env.example        # API 密钥模板（复制为 .env 填入真实 key）
├── .gitignore
├── README.md
├── 01-skills/          # 第 1 阶段：AI 基础技能
│   ├── 01_hello_ai.py           # API 调用入门：3 种调用方式
│   ├── 02_prompt_templates.py   # 提示词工程：4 种实用模板
│   └── 03_chatbot.py            # 综合实战：命令行聊天机器人
├── 02-ollama/          # 第 2 阶段：本地模型部署（待完成）
├── 03-mcp/             # 第 3 阶段：MCP 协议（待完成）
├── 04-rag/             # 第 4 阶段：RAG 检索增强生成（待完成）
├── 05-agent-skills/    # 第 5 阶段：Agent 核心能力（待完成）
├── 06-agent/           # 第 6 阶段：完整 Agent（待完成）
├── 07-multi-agent/     # 第 7 阶段：多 Agent 协作（待完成）
└── 08-projects/        # 第 8 阶段：实战项目（待完成）
```

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/你的用户名/ai-learning.git
cd ai-learning
```

### 2. 安装依赖

```bash
pip install openai python-dotenv
```

### 3. 配置 API Key

```bash
# 复制配置文件
cp .env.example .env

# 编辑 .env，填入你的 API Key
# 推荐使用 DeepSeek（国内可访问，新用户免费额度）
# 注册地址：https://platform.deepseek.com
```

### 4. 运行

```bash
# 第 1 课：API 调用入门
python 01-skills/01_hello_ai.py

# 第 2 课：提示词模板
python 01-skills/02_prompt_templates.py

# 第 3 课：聊天机器人
python 01-skills/03_chatbot.py
```

### 5. 在 VS Code 中运行

安装 Python 扩展后，打开任意 `.py` 文件，点击右上角 ▶ 按钮即可运行。或使用快捷键 `Ctrl + F5`。

## 技术栈

- **语言**：Python 3.12+
- **API 调用**：OpenAI SDK（兼容 DeepSeek / 通义千问 / Ollama）
- **框架**（待学习）：LangChain、LangGraph、LlamaIndex、FastAPI
- **向量数据库**（待学习）：Chroma、Milvus
- **本地模型**（待学习）：Ollama
- **部署**（待学习）：Docker

## 核心概念速查

| 概念 | 一句话理解 |
|------|-----------|
| Messages | 三个角色：`system`(人设) → `user`(你) → `assistant`(AI) |
| Temperature | 0=严谨，0.7=平衡，1.5=创意。代码用低温，创意用高温 |
| Streaming | `stream=True`，AI 逐字输出，用户体验好 |
| RAG | 检索相关文档 → 喂给 AI → 基于文档回答 |
| Agent | AI 自主决定用什么工具、执行什么步骤 |
| Function Calling | 让 AI 能调用你的函数/API |

## API 选择

| 服务 | 访问 | 价格 | 注册 |
|------|------|------|------|
| DeepSeek | 国内直连 | 便宜 | [platform.deepseek.com](https://platform.deepseek.com) |
| 通义千问 | 国内直连 | 便宜 | [dashscope.aliyun.com](https://dashscope.aliyun.com) |
| OpenAI | 需要代理 | 较贵 | [platform.openai.com](https://platform.openai.com) |
| Ollama | 本地免费 | 免费 | [ollama.com](https://ollama.com) |

## 学习建议

1. 每个阶段都要有能跑的代码，不只是看完文档
2. RAG 和 Agent 是核心，各花 2-3 周做深做透
3. 用 AI 学 AI，遇到不懂直接问
4. 项目要能 Docker 部署，GitHub 有完整代码和文档

## License

MIT
