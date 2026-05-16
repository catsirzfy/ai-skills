"""
MCP Client — 连接 AI 和外部工具的桥梁。

=== 核心流程 ===

用户说"帮我看看现在几点"
    → AI 分析：需要调用 get_current_time 工具
    → Client 把工具调用请求发给 MCP Server
    → Server 执行 get_current_time()，返回 "2026-05-16 10:30:00"
    → Client 把结果喂回 AI
    → AI 组织语言："现在是 5 月 16 日上午 10:30"

这就是 Agent Tool Use 的核心循环——后面学 Agent 时，底层逻辑完全一样。

=== OpenAI Function Calling 机制 ===

这是 OpenAI 定义的标准工具调用协议，DeepSeek 也兼容：
1. 在请求里传 tools 参数（工具定义列表）
2. AI 返回时可能包含 tool_calls（它想调用的工具）
3. 你执行工具，把结果以 role="tool" 的消息发回去
4. AI 基于结果生成最终回复
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI
from dotenv import load_dotenv
from mcp_server import handle_request, TOOLS

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "your-deepseek-key"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)


def call_mcp_tool(tool_name: str, arguments: dict) -> str:
    """模拟调用远程 MCP Server 的工具。

    在生产环境中，这里会通过 HTTP 或 stdio 和远程 MCP Server 通信。
    这里我们把 Server 的函数直接导入调用——协议相同，只是传输方式简化了。
    """
    request = {"method": "tools/call", "tool": tool_name, "arguments": arguments}
    response = handle_request(request)
    return response.get("result", response.get("error", "未知错误"))


def chat_with_tools(user_message: str):
    """一次完整的"AI + 工具调用"对话。

    这是 Agent 循环的最小实现（简化版 ReAct）：
    1. 把用户消息和可用工具列表发给 AI
    2. AI 返回 text 或 tool_call
    3. 如果是 tool_call → 执行工具 → 把结果发回 AI → 得到最终回复
    4. 如果是 text → 直接返回

    真正的 Agent 会循环 2-3 步多次，直到完成任务。
    """
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个能使用工具的 AI 助手。"
                "当需要实时信息、读取文件或执行代码时，请调用相应的工具。"
            ),
        },
        {"role": "user", "content": user_message},
    ]

    # --- 把 TOOLS 定义转换为 OpenAI Function Calling 格式 ---
    # OpenAI 要求的工具格式和 MCP 略有不同，这里做转换
    openai_tools = []
    for t in TOOLS:
        params = {}
        required = []
        for name, info in t.get("parameters", {}).items():
            params[name] = {"type": info["type"], "description": info["description"]}
            required.append(name)
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": {
                    "type": "object",
                    "properties": params,
                    "required": required,
                },
            },
        })

    # --- 第一次调用：AI 决定是否需要工具 ---
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=openai_tools,  # 告诉 AI 有哪些工具可用
    )

    msg = response.choices[0].message

    # 如果 AI 没有调用工具，直接返回文本回复
    if not msg.tool_calls:
        print(f"AI: {msg.content}")
        return msg.content

    # --- AI 请求了工具调用 — 执行工具 ---
    # 把 AI 的 tool_calls 消息加入对话历史
    messages.append(msg.model_dump())

    for tool_call in msg.tool_calls:
        tool_name = tool_call.function.name
        # AI 返回的 arguments 是 JSON 字符串，需要解析
        arguments = json.loads(tool_call.function.arguments)

        print(f"\n[Tool] 调用: {tool_name}({arguments})")
        result = call_mcp_tool(tool_name, arguments)
        print(f"[Tool] 结果: {result[:200]}{'...' if len(result) > 200 else ''}")

        # 把工具执行结果加入对话历史
        # role="tool" 告诉 AI："这是刚才那个工具调用的返回结果"
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })

    # --- 第二次调用：AI 基于工具结果生成最终回答 ---
    final_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
    )
    final_answer = final_response.choices[0].message.content
    print(f"\nAI: {final_answer}")
    return final_answer


def run_demo():
    """用 4 个例子演示 MCP 工具调用的完整流程。"""
    examples = [
        "现在几点了？",
        "帮我算一下 12345 * 67890 等于多少",
        "列出 D:\\实验 目录下有什么文件",
        "读取 D:\\实验\\PythonAI应用开发学习路线.md 文件的前几行内容",
    ]

    for question in examples:
        print(f"\n{'=' * 60}")
        print(f"用户: {question}")
        print("=" * 60)
        try:
            chat_with_tools(question)
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    if client.api_key == "your-deepseek-key":
        print("未配置 API Key，无法运行 Demo。")
        print("cp .env.example .env  # 然后编辑 .env 填入你的 key")
        sys.exit(0)
    run_demo()
