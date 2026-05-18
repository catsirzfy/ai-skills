"""
Agent Skill 4：流式输出（Streaming）

=== 为什么 Agent 需要流式输出 ===

Agent 执行时用户可能等很久（搜索 → 调工具 → 生成），流式输出让用户看到进度：
- "正在搜索知识库..."  → 知道在干什么
- "找到 3 条相关文档"   → 看到中间结果
- 逐字输出最终回答      → 不用等全部完成

=== 两种流式场景 ===

1. LLM 回答流式：AI 思考时逐字显示（打字机效果）
2. Agent 过程流式：每一步工具调用都实时反馈（进度条效果）

两者结合 = 用户体验最好的 Agent 交互方式。
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
# 场景 1：纯流式回答（已在 Stage 1 学过，这里快速复习）
# ================================================================
def stream_answer(prompt: str):
    """流式输出 AI 的思考过程——用户看到文字一个个出现。"""
    print("AI: ", end="", flush=True)
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    full = ""
    for chunk in stream:
        if content := chunk.choices[0].delta.content:
            print(content, end="", flush=True)
            full += content
    print()
    return full


# ================================================================
# 场景 2：Agent 过程流式（Step-by-step 可视化）
# ================================================================
def get_weather(city: str) -> str:
    w = {"北京": "晴 25°C", "上海": "多云 28°C", "深圳": "阵雨 30°C", "杭州": "阴 22°C"}
    return w.get(city, f"{city}: 晴 24°C")


def search_knowledge(query: str) -> str:
    kb = {"rag": "RAG是检索增强生成...", "agent": "AI Agent是自主智能体..."}
    for k, v in kb.items():
        if k in query.lower():
            return v
    return f"未找到 '{query}'"


TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "查天气", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "search_knowledge", "description": "搜知识", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]
TOOL_MAP = {"get_weather": get_weather, "search_knowledge": search_knowledge}


def streaming_agent(question: str):
    """Agent + 流式：每一步都实时展示。

    关键设计：工具调用用 print 显示进度，最终回答用流式逐字输出。
    这样用户知道 Agent 在干什么，不会觉得卡住了。
    """
    print(f"用户: {question}\n")
    messages = [
        {"role": "system", "content": "你是一个能使用工具的助手。当需要查天气或搜索知识时，使用对应工具。"},
        {"role": "user", "content": question},
    ]

    # Agent 循环：工具调用阶段
    for iteration in range(3):
        response = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        msg = response.choices[0].message

        if not msg.tool_calls:
            # 最终回答阶段 — 流式输出
            messages.append({"role": "assistant", "content": msg.content or ""})
            break

        messages.append(msg.model_dump())

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"  🔍 Agent 正在调用: {name}({args})...")
            result = TOOL_MAP[name](**args)
            print(f"  ✅ {name} 返回: {result}\n")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # 最终答案 — 流式逐字输出
    print("AI: ", end="", flush=True)
    stream = client.chat.completions.create(model=MODEL, messages=messages, stream=True)
    full = ""
    for chunk in stream:
        if content := chunk.choices[0].delta.content:
            print(content, end="", flush=True)
            full += content
    print("\n")
    return full


# ================================================================
# 场景 3：SSE 格式流式（FastAPI 生产环境标准）
# ================================================================
def sse_stream_demo(prompt: str):
    """演示 SSE（Server-Sent Events）格式的流式输出。

    这就是 api_server.py 里 /chat/stream 端点使用的格式。
    前端 JavaScript 用 EventSource API 接收这些数据。
    """
    print("=== SSE 格式（前端用 EventSource 接收）===\n")

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        if content := chunk.choices[0].delta.content:
            # SSE 格式：data: JSON\n\n
            event = json.dumps({"text": content}, ensure_ascii=False)
            print(f"data: {event}")
    print("data: [DONE]")
    print("\n前端代码示例：")
    print("  const evtSource = new EventSource('/chat/stream');")
    print("  evtSource.onmessage = (e) => { if (e.data === '[DONE]') return; showText(JSON.parse(e.data).text); };")


if __name__ == "__main__":
    print("=" * 60)
    print("场景 1：纯流式回答")
    print("=" * 60)
    stream_answer("用一句话介绍什么是 Python 的上下文管理器")

    print("\n" + "=" * 60)
    print("场景 2：Agent 过程流式（Step-by-step）")
    print("=" * 60)
    streaming_agent("帮我查一下北京的天气，再告诉我什么是 Agent")

    print("=" * 60)
    print("场景 3：SSE 格式流式")
    print("=" * 60)
    sse_stream_demo("用一句话介绍 FastAPI")
