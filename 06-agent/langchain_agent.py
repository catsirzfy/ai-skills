"""
LangChain Agent — 自主决策 + 多工具调用的智能体。

=== Agent 架构 ===

感知 → 思考 → 行动 → 观察 → 循环

LangChain 的 Agent 封装了这个循环：
- 你只需定义工具 + LLM + 提示词
- LangChain 自动执行 ReAct 循环
- 大幅减少代码量

=== 本文件演示 ===

配备 6 个工具的 Agent：
1. 搜索知识库     2. 计算器       3. 获取时间
4. 读取文件       5. 列出目录     6. 执行 Python 代码

这 6 个工具覆盖了 Agent 最常见的三类能力：
- 信息获取（搜索、读文件、列目录、时间）
- 计算推理（计算器）
- 代码执行（Python）
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

# ================================================================
# 工具定义
# ================================================================
import json
import math
from datetime import datetime
from pathlib import Path


def get_current_time() -> str:
    """获取当前日期时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculator(expression: str) -> str:
    """安全计算数学表达式。支持 + - * / ** sqrt() 等。"""
    allowed = {"__builtins__": {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow, "sqrt": math.sqrt, "len": len,
        "int": int, "float": float, "str": str, "list": list,
        "True": True, "False": False, "None": None,
    }}
    try:
        result = eval(expression, allowed, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


def search_knowledge(query: str) -> str:
    """在本地知识库中搜索技术概念。"""
    kb = {
        "python": "Python 1991年发布，解释型面向对象语言，以简洁易读著称。",
        "fastapi": "FastAPI 高性能 Python Web 框架，支持异步、自动生成 API 文档。",
        "rag": "RAG(检索增强生成)结合信息检索和文本生成，是 AI 应用核心方向。",
        "agent": "AI Agent 是能自主感知、决策、行动的智能体系统，Function Calling 是基础。",
        "langgraph": "LangGraph 是 LangChain 的 Agent 框架，用状态图定义 Agent 工作流。",
        "langchain": "LangChain LLM 应用开发框架，组件化设计，支持 Chain 和 Agent。",
        "embedding": "Embedding 将文本转为向量，用于语义搜索和相似度计算。",
        "vector db": "向量数据库专门存储和检索向量，如 Chroma、Milvus、Pinecone。",
    }
    for key, value in kb.items():
        if key in query.lower():
            return value
    return f"知识库中暂未找到关于 '{query}' 的信息。"


def read_file(path: str) -> str:
    """读取文件内容。"""
    try:
        p = Path(path)
        if not p.exists():
            return f"文件不存在: {path}"
        content = p.read_text(encoding="utf-8")
        if len(content) > 2000:
            content = content[:2000] + "\n...(文件过长，已截断)"
        return content
    except Exception as e:
        return f"读取失败: {e}"


def list_directory(path: str) -> str:
    """列出目录内容。"""
    try:
        p = Path(path)
        if not p.is_dir():
            return f"不是有效目录: {path}"
        items = []
        for item in sorted(p.iterdir()):
            prefix = "[DIR]" if item.is_dir() else "[FILE]"
            items.append(f"{prefix} {item.name}")
        return "\n".join(items) if items else "(空目录)"
    except Exception as e:
        return f"列出失败: {e}"


def execute_python(code: str) -> str:
    """在沙箱中执行 Python 代码（仅限简单操作，生产环境用 Docker）。"""
    import io
    try:
        buf = io.StringIO()
        _stdout = sys.stdout
        sys.stdout = buf
        try:
            exec(code, {"__builtins__": {
                "print": print, "range": range, "len": len, "sum": sum,
                "max": max, "min": min, "sorted": sorted, "list": list,
                "dict": dict, "set": set, "str": str, "int": int, "float": float,
                "bool": bool, "abs": abs, "round": round, "enumerate": enumerate,
                "zip": zip, "map": map, "filter": filter,
            }})
        finally:
            sys.stdout = _stdout
        output = buf.getvalue().strip()
        return output if output else "(无输出)"
    except Exception as e:
        return f"执行错误: {e}"


# ================================================================
# LangChain Agent
# ================================================================
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# 用 @tool 装饰器把 Python 函数变成 LangChain 工具
# 函数名 = 工具名, docstring = description, 参数注解 = parameters
# 比手写 JSON 定义方便很多

@tool
def lc_get_current_time() -> str:
    """获取当前日期和时间。当用户问现在几点、今天日期时使用。"""
    return get_current_time()


@tool
def lc_calculator(expression: str) -> str:
    """执行数学计算。当用户需要算数、数学表达式时使用。参数是要计算的表达式，如 '123 * 456'。"""
    return calculator(expression)


@tool
def lc_search_knowledge(query: str) -> str:
    """搜索技术知识库。当用户问技术概念的定义、解释时使用。参数是搜索关键词。"""
    return search_knowledge(query)


@tool
def lc_read_file(path: str) -> str:
    """读取文件内容。当用户想看某个文件的内容时使用。参数是文件路径。"""
    return read_file(path)


@tool
def lc_list_directory(path: str) -> str:
    """列出目录下的文件和文件夹。当用户想看目录内容时使用。参数是目录路径。"""
    return list_directory(path)


@tool
def lc_execute_python(code: str) -> str:
    """执行 Python 代码。当用户需要运行代码、处理数据时使用。参数是要执行的 Python 代码。"""
    return execute_python(code)


TOOL_LIST = [
    lc_get_current_time, lc_calculator, lc_search_knowledge,
    lc_read_file, lc_list_directory, lc_execute_python,
]


def build_agent():
    """构建 LangChain Agent。

    create_tool_calling_agent + AgentExecutor：
    - create_tool_calling_agent：创建能调用工具的 Agent（ReAct 循环）
    - AgentExecutor：执行 Agent 循环，包括错误处理和重试
    """
    from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY", "your-deepseek-key"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0.3,
    )

    # Agent 的 system prompt — 定义 Agent 的行为规范
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "你是一个智能助手 Agent，能使用工具完成任务。"
            "规则："
            "1. 当需要获取信息时，使用对应的搜索或文件工具"
            "2. 当需要计算时，使用计算器工具"
            "3. 当需要执行代码时，使用 Python 执行工具"
            "4. 如果不能完成任务，如实告知"
            "5. 使用中文回答"
        )),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, TOOL_LIST, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=TOOL_LIST,
        verbose=True,           # True = 打印 Agent 的思考过程
        max_iterations=5,       # 最多 5 步
        handle_parsing_errors=True,  # 自动处理 JSON 解析错误
    )
    return executor


if __name__ == "__main__":
    agent = build_agent()

    tasks = [
        "帮我计算 (12345 * 67890 + 5000) / 2 的结果",
        "现在几点了？",
        "什么是 LangGraph？",
        "列出 D:\\实验 目录下有哪些文件",
        "用 Python 输出斐波那契数列的前 10 个数",
    ]

    for task in tasks:
        print(f"\n{'=' * 60}")
        print(f"任务: {task}")
        print("=" * 60)
        result = agent.invoke({"input": task})
        print(f"\n最终结果: {result['output']}\n")
