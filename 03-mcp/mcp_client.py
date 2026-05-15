"""MCP Client — connects AI to external tools via MCP Server.

Flow:
    User asks question → AI decides it needs a tool → Client calls MCP Server
    → Tool executes → Result goes back to AI → AI formulates final response

This is the foundation of Agent tool-use. Every Agent framework works this way.
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
    """Simulate calling an MCP server tool.

    In production, this would be a network call to a separate MCP server process.
    Here we inline it for clarity — the protocol is identical.
    """
    request = {"method": "tools/call", "tool": tool_name, "arguments": arguments}
    response = handle_request(request)
    return response.get("result", response.get("error", "Unknown error"))


def chat_with_tools(user_message: str):
    """Single conversation turn with automatic tool use.

    This is the core Agent loop simplified:
    1. Send user message + tool definitions to AI
    2. AI responds with either text OR a tool_call
    3. If tool_call → execute it → feed result back to AI → AI gives final answer
    """
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个能使用工具的 AI 助手。"
                "当需要实时信息、读取文件、执行代码时，使用提供的工具。"
                "Tool calling instructions:\n"
                "- get_current_time: no args needed\n"
                "- read_file: path=<string>\n"
                "- list_directory: path=<string>\n"
                "- execute_python: code=<string>\n"
            ),
        },
        {"role": "user", "content": user_message},
    ]

    # tools definition in OpenAI-compatible format
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

    # First call — AI decides whether to use a tool
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=openai_tools,
    )

    msg = response.choices[0].message

    # If AI doesn't need a tool, just return its response
    if not msg.tool_calls:
        print(f"AI (no tool): {msg.content}")
        return msg.content

    # AI requested tools — execute them
    messages.append(msg.model_dump())

    for tool_call in msg.tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        print(f"\n[Tool] Calling: {tool_name}({arguments})")
        result = call_mcp_tool(tool_name, arguments)
        print(f"[Tool] Result: {result[:200]}{'...' if len(result) > 200 else ''}")

        # Append tool result to conversation
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })

    # Second call — AI synthesizes final answer with tool results
    final_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
    )
    final_answer = final_response.choices[0].message.content
    print(f"\nAI (final): {final_answer}")
    return final_answer


def run_demo():
    """Demonstrate MCP tool-use with a few examples."""
    examples = [
        "现在几点了？",
        "帮我算一下 12345 * 67890 等于多少",
        "列出 D:\\实验 目录下有什么文件",
        "读取 D:\\实验\\PythonAI应用开发学习路线.md 文件的前5行",
    ]

    for question in examples:
        print(f"\n{'=' * 60}")
        print(f"User: {question}")
        print("=" * 60)
        try:
            chat_with_tools(question)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    if client.api_key == "your-deepseek-key":
        print("No API key configured. Skipping demo.")
        print("cp .env.example .env  # then edit .env with your key")
        sys.exit(0)
    run_demo()
