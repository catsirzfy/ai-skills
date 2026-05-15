"""MCP Server — exposes tools to AI via standard protocol.

Architecture:
    AI Model → MCP Client → MCP Server → Tools (files, DB, APIs)

This server exposes: read_file, list_directory, get_current_time, execute_python.
"""

import os
import json
import sys
import subprocess
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------
# Tool definitions (the "contract" AI sees)
# ---------------------------------------------------------------
TOOLS = [
    {
        "name": "get_current_time",
        "description": "Get the current date and time.",
        "parameters": {},
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file at the given path.",
        "parameters": {
            "path": {"type": "string", "description": "Absolute path to the file."}
        },
    },
    {
        "name": "list_directory",
        "description": "List files and folders in a directory.",
        "parameters": {
            "path": {"type": "string", "description": "Absolute path to the directory."}
        },
    },
    {
        "name": "execute_python",
        "description": "Execute a Python expression and return the result.",
        "parameters": {
            "code": {"type": "string", "description": "Python code to execute via exec()."}
        },
    },
]


# ---------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------
def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: file not found — {path}"
    except PermissionError:
        return f"Error: permission denied — {path}"


def list_directory(path: str) -> str:
    try:
        entries = os.listdir(path)
        return "\n".join(
            f"{'[DIR] ' if os.path.isdir(os.path.join(path, e)) else '[FILE]'} {e}"
            for e in sorted(entries)
        )
    except FileNotFoundError:
        return f"Error: directory not found — {path}"
    except PermissionError:
        return f"Error: permission denied — {path}"


def execute_python(code: str) -> str:
    import io
    try:
        buf = io.StringIO()
        _stdout = sys.stdout
        sys.stdout = buf
        try:
            exec(code, {})
        finally:
            sys.stdout = _stdout
        output = buf.getvalue().strip()
        return output if output else "(no output)"
    except Exception as e:
        return f"Error: {e}"


# Tool router
TOOL_MAP = {
    "get_current_time": get_current_time,
    "read_file": read_file,
    "list_directory": list_directory,
    "execute_python": execute_python,
}


# ---------------------------------------------------------------
# MCP Server (simplified — JSON-RPC via stdin/stdout)
# ---------------------------------------------------------------
def handle_request(request: dict) -> dict:
    """Process a single MCP JSON-RPC request."""
    method = request.get("method")

    if method == "tools/list":
        return {"tools": TOOLS}

    if method == "tools/call":
        tool_name = request.get("tool")
        arguments = request.get("arguments", {})

        func = TOOL_MAP.get(tool_name)
        if not func:
            return {"error": f"Unknown tool: {tool_name}"}

        result = func(**arguments)
        return {"result": result}

    return {"error": f"Unknown method: {method}"}


def run_stdio_server():
    """Run MCP server over stdin/stdout (JSON-RPC)."""
    print("MCP Server ready (stdin/stdout mode)", file=sys.stderr)
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            request = json.loads(line.strip())
            response = handle_request(request)
            print(json.dumps(response, ensure_ascii=False), flush=True)
        except json.JSONDecodeError:
            print(json.dumps({"error": "Invalid JSON"}), flush=True)
        except EOFError:
            break


if __name__ == "__main__":
    run_stdio_server()
