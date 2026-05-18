"""
Agent Skill 1：工具调用（Tool Use / Function Calling）

=== 为什么这是 Agent 的基础 ===

纯 LLM 只能"说"，不能"做"。Function Calling 让 LLM 能：
- 查数据库 → 调用 SQL 函数
- 看天气   → 调用天气 API
- 算数学   → 调用计算函数
- 发邮件   → 调用邮件 API

Agent = LLM + 工具调用 + 自主决策循环。没有工具调用，就没有 Agent。

=== 机制 ===

1. 你定义函数 + 描述（name, description, parameters）
2. 把函数描述告诉 AI（tools 参数）
3. AI 决定：要不要调工具？调哪个？传什么参数？
4. 你执行函数，把结果返回给 AI
5. AI 基于结果生成最终回答
"""

import os
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "your-deepseek-key"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)
MODEL = "deepseek-chat"


# ================================================================
# 定义工具 — 就像给 AI 一个"工具箱"
# ================================================================

def get_weather(city: str) -> str:
    """查询指定城市的天气。"""
    # 模拟天气数据（生产环境这里调真实 API）
    weather_db = {
        "北京": "晴，25°C，湿度 40%",
        "上海": "多云，28°C，湿度 65%",
        "深圳": "阵雨，30°C，湿度 80%",
        "杭州": "阴，22°C，湿度 55%",
    }
    return weather_db.get(city, f"{city}：晴，24°C，湿度 50%")


def calculate(expression: str) -> str:
    """执行数学计算。支持加减乘除、乘方、括号等。"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


def search_knowledge(query: str) -> str:
    """在知识库中搜索信息。"""
    kb = {
        "python": "Python 是一种解释型、面向对象的高级编程语言，由 Guido van Rossum 于 1991 年发布。",
        "rag": "RAG（检索增强生成）是一种结合信息检索和文本生成的 AI 技术架构。",
        "agent": "AI Agent 是能自主感知环境、做出决策、执行行动的智能体系统。",
        "langchain": "LangChain 是一个用于构建 LLM 应用的框架，提供组件化和链式调用能力。",
    }
    for key, value in kb.items():
        if key in query.lower():
            return value
    return f"未找到关于 '{query}' 的信息"


# 工具定义表 — 告诉 AI 每个工具干什么、需要什么参数
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气状况。当用户询问天气时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，如 北京、上海、杭州"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算。当用户需要算数时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 123 * 456 + 789"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "在知识库中搜索技术概念。当用户询问 Python、RAG、Agent 等概念时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"],
            },
        },
    },
]

# 工具名 → 实际函数的映射
TOOL_MAP = {"get_weather": get_weather, "calculate": calculate, "search_knowledge": search_knowledge}


# ================================================================
# Agent 循环 — AI 自主决定用什么工具
# ================================================================
def agent_loop(user_message: str, max_iterations: int = 5) -> str:
    """一次 Agent 对话：AI 可以多次调用工具直到得到最终答案。

    关键：这个循环就是 Agent 的核心——
    1. 发给 AI（含工具列表）
    2. AI 返回 text → 结束（最终答案）
    3. AI 返回 tool_call → 执行工具 → 结果回传 → 回到第 1 步
    """
    messages = [
        {"role": "system", "content": "你是一个能使用工具的 AI 助手。当需要查询天气、执行计算或搜索知识时，请调用相应的工具。"},
        {"role": "user", "content": user_message},
    ]

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS
        )
        msg = response.choices[0].message

        # 没有工具调用 → AI 给出了最终答案
        if not msg.tool_calls:
            print(f"AI 回答: {msg.content}\n")
            return msg.content

        # 有工具调用 → 执行
        messages.append(msg.model_dump())

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            print(f"  🔧 调用工具: {name}({args})")
            result = TOOL_MAP[name](**args)
            print(f"  📋 工具返回: {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return "达到最大迭代次数，任务未完成。"


if __name__ == "__main__":
    tests = [
        "帮我查一下北京的天气怎么样",
        "12345 乘以 67890 再减去 1000 等于多少？",
        "什么是 Agent？",
        "上海的天气如何？顺便告诉我 2 的 10 次方是多少",  # 需要两个工具
    ]

    for q in tests:
        print(f"{'=' * 60}")
        print(f"用户: {q}")
        print("=" * 60)
        agent_loop(q)
