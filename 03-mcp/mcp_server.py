"""
MCP Server — 通过标准协议向 AI 暴露工具能力。

=== MCP 是什么 ===

MCP（Model Context Protocol）是 Anthropic 提出的开放标准，让 AI 模型
能通过统一协议调用外部工具和数据源。

在 MCP 之前，每个 AI 框架都有自己的工具调用方式（LangChain Tool、OpenAI Function 等）。
MCP 的目标是统一这些——就像 USB 统一外设接口一样。

=== 架构 ===

    AI Model → MCP Client → MCP Server → 实际工具（文件系统/数据库/API）

    1. MCP Server 暴露工具列表（tools/list）
    2. AI 决定调用哪个工具（tools/call）
    3. Server 执行并返回结果
    4. AI 基于结果生成最终回复

=== 本文件实现 ===

这是一个简化的 MCP Server，通过 stdin/stdout 的 JSON-RPC 通信。
生产环境中，MCP 可以通过 HTTP/SSE 或 stdio 等多种传输方式运行。

暴露的工具：读文件、列目录、获取时间、执行 Python 代码
"""

import os
import json
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ================================================================
# 工具定义 — 这是 AI 看到的"工具菜单"
# ================================================================
# 每个工具包含 name、description、parameters。
# AI 根据这些信息决定：用户的问题需要用哪个工具？
# description 写得好不好，直接影响 AI 选择工具的正确率。
TOOLS = [
    {
        "name": "get_current_time",
        "description": "获取当前日期和时间。",
        "parameters": {},  # 无参数
    },
    {
        "name": "read_file",
        "description": "读取指定路径的文件内容。",
        "parameters": {
            "path": {"type": "string", "description": "文件的绝对路径。"}
        },
    },
    {
        "name": "list_directory",
        "description": "列出目录下的文件和文件夹。",
        "parameters": {
            "path": {"type": "string", "description": "目录的绝对路径。"}
        },
    },
    {
        "name": "execute_python",
        "description": "执行一段 Python 代码并捕获输出。",
        "parameters": {
            "code": {"type": "string", "description": "要执行的 Python 代码。"}
        },
    },
]


# ================================================================
# 工具实现 — 真正干活的函数
# ================================================================
def get_current_time() -> str:
    """返回当前时间的格式化字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_file(path: str) -> str:
    """读文件，处理常见的错误情况。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"错误: 文件不存在 — {path}"
    except PermissionError:
        return f"错误: 没有权限读取 — {path}"


def list_directory(path: str) -> str:
    """列目录，区分文件和文件夹。"""
    try:
        entries = os.listdir(path)
        return "\n".join(
            f"{'[DIR] ' if os.path.isdir(os.path.join(path, e)) else '[FILE]'} {e}"
            for e in sorted(entries)
        )
    except FileNotFoundError:
        return f"错误: 目录不存在 — {path}"
    except PermissionError:
        return f"错误: 没有权限访问 — {path}"


def execute_python(code: str) -> str:
    """安全执行 Python 代码（捕获 stdout）。

    注意：生产环境中需要对代码做沙箱隔离（Docker/subprocess），
    直接 exec() 用户输入的代码有安全风险。
    """
    import io
    try:
        buf = io.StringIO()
        _stdout = sys.stdout
        sys.stdout = buf  # 重定向 stdout，捕获 print 输出
        try:
            exec(code, {})
        finally:
            sys.stdout = _stdout  # 恢复 stdout
        output = buf.getvalue().strip()
        return output if output else "(无输出)"
    except Exception as e:
        return f"错误: {e}"


# 工具名 → 实现函数的映射
TOOL_MAP = {
    "get_current_time": get_current_time,
    "read_file": read_file,
    "list_directory": list_directory,
    "execute_python": execute_python,
}


# ================================================================
# MCP 协议处理 — JSON-RPC over stdin/stdout
# ================================================================
def handle_request(request: dict) -> dict:
    """处理单个 MCP JSON-RPC 请求。

    支持两个方法：
    - tools/list:  返回所有可用工具的列表
    - tools/call:  调用指定工具并返回结果
    """
    method = request.get("method")

    if method == "tools/list":
        # AI 启动时先问"你有什么工具？"
        return {"tools": TOOLS}

    if method == "tools/call":
        # AI 决定用某个工具时调用
        tool_name = request.get("tool")
        arguments = request.get("arguments", {})

        func = TOOL_MAP.get(tool_name)
        if not func:
            return {"error": f"未知工具: {tool_name}"}

        # **arguments 把参数字典展开为关键字参数
        result = func(**arguments)
        return {"result": result}

    return {"error": f"未知方法: {method}"}


def run_stdio_server():
    """通过 stdin/stdout 运行 MCP Server。

    每行是一个完整的 JSON-RPC 请求（请求），
    回复一行 JSON（响应）。
    """
    print("MCP Server 已就绪（stdin/stdout 模式）", file=sys.stderr)
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            request = json.loads(line.strip())
            response = handle_request(request)
            # flush=True 确保响应立即输出，客户端不会一直等待
            print(json.dumps(response, ensure_ascii=False), flush=True)
        except json.JSONDecodeError:
            print(json.dumps({"error": "JSON 格式错误"}), flush=True)
        except EOFError:
            break


if __name__ == "__main__":
    run_stdio_server()
