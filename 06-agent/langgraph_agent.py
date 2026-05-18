"""
LangGraph Agent — 用状态图定义 Agent 工作流。

=== LangGraph vs LangChain Agent ===

LangChain Agent：封装好的黑盒，ReAct 循环自动运行。简单但不够灵活。
LangGraph：让你用图（Graph）定义 Agent 的每一步和流转条件。更灵活但代码更多。

=== 为什么学 LangGraph ===

面试高频！LangGraph 是目前最主流的 Agent 框架：
- 用 State 对象保存 Agent 的完整状态
- 用 Node 定义每个处理步骤
- 用 Edge 定义步骤之间的流转条件
- 可视化、可调试、可定制

=== 本文件演示 ===

一个完整的 LangGraph Agent：
- 两个工具节点：搜索 + 计算
- 条件路由：AI 决定下一步是调工具还是结束
- 状态持久化：可以看到 Agent 每一步的状态变化
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "your-deepseek-key"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)
MODEL = "deepseek-chat"

# ================================================================
# 工具
# ================================================================
import math
import json
from datetime import datetime
from pathlib import Path


def search_knowledge(query: str) -> str:
    kb = {
        "python": "Python 1991年发布，解释型面向对象语言。",
        "rag": "RAG 检索增强生成，结合信息检索和文本生成。",
        "agent": "AI Agent 自主智能体，能感知、决策、执行。",
        "langgraph": "LangGraph 用状态图定义 Agent 工作流。",
        "fastapi": "FastAPI 高性能 Python Web 框架。",
    }
    for k, v in kb.items():
        if k in query.lower():
            return v
    return f"未找到关于 '{query}' 的信息"


def calculator(expression: str) -> str:
    allowed = {"__builtins__": {"abs": abs, "round": round, "sqrt": math.sqrt, "pow": pow, "int": int, "float": float}}
    try:
        return f"{expression} = {eval(expression, allowed, {})}"
    except Exception as e:
        return f"计算错误: {e}"


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_file(path: str) -> str:
    try:
        content = Path(path).read_text(encoding="utf-8")
        return content[:2000]
    except Exception as e:
        return f"读文件失败: {e}"


# ================================================================
# 工具定义（Function Calling 格式）
# ================================================================
TOOLS = [
    {"type": "function", "function": {"name": "search", "description": "搜索知识库", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "calculate", "description": "数学计算", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_time", "description": "获取当前时间", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "read_file", "description": "读取文件", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
]
TOOL_MAP = {"search": search_knowledge, "calculate": calculator, "get_time": get_current_time, "read_file": read_file}


# ================================================================
# LangGraph 核心：状态图
# ================================================================
from typing_extensions import TypedDict, Annotated
from typing import Literal
import operator


class AgentState(TypedDict):
    """Agent 的状态 — 这是 LangGraph 的核心概念。

    所有节点共享这个状态对象，节点可以读取和修改状态。
    messages 用 operator.add 合并，意味着新消息会追加到列表末尾。
    """
    messages: Annotated[list[dict], operator.add]  # 对话历史
    iteration: int                                 # 当前步数
    final_answer: str                              # 最终答案


# 节点 1：Agent 思考节点
def agent_node(state: AgentState) -> dict:
    """Agent 的核心——调用 LLM 决定下一步做什么。

    返回值会合并到 state 中（因为 messages 标记了 operator.add）。
    """
    iteration = state.get("iteration", 0)
    messages = state.get("messages", [])

    response = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOLS,
    )
    msg = response.choices[0].message

    if msg.tool_calls:
        # 把 tool_calls 消息加入历史
        result = {"messages": [msg.model_dump()], "iteration": iteration + 1}
    else:
        result = {
            "messages": [msg.model_dump()],
            "iteration": iteration + 1,
            "final_answer": msg.content or "",
        }
    return result


# 节点 2：工具执行节点
def tool_node(state: AgentState) -> dict:
    """执行 AI 请求的工具调用，返回结果。"""
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else {}

    tool_messages = []
    for tc in last_msg.get("tool_calls", []):
        name = tc["function"]["name"]
        args = json.loads(tc["function"]["arguments"])
        print(f"  🔧 {name}({args})")
        result = TOOL_MAP[name](**args)
        print(f"  📋 -> {result[:80]}")
        tool_messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": result,
        })

    return {"messages": tool_messages}


# 路由函数：决定下一步去哪个节点
def router(state: AgentState) -> Literal["tool_node", "__end__"]:
    """根据 AI 的响应决定流转方向。

    - AI 调用了工具 → 去 tool_node
    - AI 给出了最终答案 → 结束
    - 超过 5 步 → 强制结束
    """
    if state.get("final_answer"):
        return "__end__"
    if state.get("iteration", 0) >= 5:
        return "__end__"

    messages = state.get("messages", [])
    if messages and messages[-1].get("tool_calls"):
        return "tool_node"
    return "__end__"


# 节点 3：工具执行后路由（回到 agent 还是结束）
def router_after_tools(state: AgentState) -> Literal["agent_node", "__end__"]:
    """工具执行完成后，回到 agent 继续思考。"""
    if state.get("iteration", 0) >= 5:
        return "__end__"
    return "agent_node"


def build_graph():
    """构建 LangGraph 工作流。

    图结构：
        agent_node ←──── router_after_tools ──── tool_node
            │                                        ↑
            │ (router → tool_node)                   │
            └────────────────────────────────────────┘
            │
            └── (router → __end__) → 输出最终答案
    """
    from langgraph.graph import StateGraph, END

    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("agent_node", agent_node)
    workflow.add_node("tool_node", tool_node)

    # 设置入口
    workflow.set_entry_point("agent_node")

    # 添加条件边：agent 节点 → 工具 or 结束
    workflow.add_conditional_edges("agent_node", router, {"tool_node": "tool_node", "__end__": END})

    # 工具节点 → 回到 agent（或者结束）
    workflow.add_conditional_edges("tool_node", router_after_tools, {"agent_node": "agent_node", "__end__": END})

    return workflow.compile()


if __name__ == "__main__":
    graph = build_graph()

    tasks = [
        "帮我算一下 2 的 20 次方，然后告诉我这个数字在计算机领域有什么意义",
        "现在几点了？顺便帮我搜索一下什么是 LangGraph",
    ]

    for task in tasks:
        print(f"\n{'=' * 60}")
        print(f"任务: {task}")
        print("=" * 60)

        initial_state = {
            "messages": [
                {"role": "system", "content": "你是一个智能 Agent。使用工具完成任务，用中文回答。"},
                {"role": "user", "content": task},
            ],
            "iteration": 0,
            "final_answer": "",
        }

        # stream 模式可以看到每一步的状态变化
        for step_output in graph.stream(initial_state):
            node_name = list(step_output.keys())[0]
            # 只打印关键信息
            state = step_output[node_name]
            if state.get("final_answer"):
                print(f"\n✅ 最终答案: {state['final_answer']}")
            print(f"  (节点: {node_name}, 步数: {state.get('iteration', 0)})")
