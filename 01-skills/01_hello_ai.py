"""
第1课：第一次调用 AI API
目标：用 Python 调用 OpenAI 兼容的 API，让 AI 回复你的消息

核心概念：
- API Key：你的身份凭证
- Model：你调用的模型名称
- Messages：你和 AI 的对话历史
- Temperature：控制回答的随机性（0=确定，1=创意）

国内可用的 API（都兼容 OpenAI 格式）：
- DeepSeek：platform.deepseek.com   推荐，便宜好用
- 通义千问：dashscope.aliyun.com
- 本地 Ollama：免费，无限调用（第4节）
"""

import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
load_dotenv()

# ============================================================
# 1. 创建客户端（连接 AI 服务）
# ============================================================
# 默认使用 DeepSeek API（国内可访问）
# 注册获取 key: https://platform.deepseek.com/api_keys
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "your-deepseek-key"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

# 如果你想用 OpenAI 官方（需要代理）：
# client = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     base_url="https://api.openai.com/v1",
# )

# 如果你想用通义千问：
# client = OpenAI(
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )

# 本地 Ollama（免费无限调用，第2阶段会讲）：
# client = OpenAI(
#     api_key="ollama",
#     base_url="http://localhost:11434/v1",
# )


# ============================================================
# 2. 最简单的调用
# ============================================================
def simple_chat():
    """发送一条消息，获取 AI 回复"""
    response = client.chat.completions.create(
        model="deepseek-chat",  # DeepSeek 的模型名
        messages=[
            {"role": "system", "content": "你是一个友好的助手。"},
            {"role": "user", "content": "用一句话介绍什么是 Python。"},
        ],
        temperature=0.7,  # 0=严谨, 1=有创意
    )

    reply = response.choices[0].message.content
    print(f"AI 回复: {reply}")

    usage = response.usage
    print(f"Token 消耗: 输入={usage.prompt_tokens}, 输出={usage.completion_tokens}")


# ============================================================
# 3. Messages 的三个角色
# ============================================================
def explain_roles():
    """理解 system / user / assistant 三个角色的区别"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个只会用古诗回答问题的诗人。"},
            {"role": "user", "content": "今天天气真好"},
        ],
    )
    print(f"诗人回答: {response.choices[0].message.content}")


# ============================================================
# 4. 流式输出（像打字机一样逐字显示）
# ============================================================
def streaming_chat():
    """流式输出，让用户看到 AI 逐字生成"""
    print("AI 正在打字: ", end="", flush=True)

    stream = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "写一首关于编程的五言诗"}],
        stream=True,
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
    print()


# ============================================================
# 5. 无需 API Key 也能运行的学习模式
# ============================================================
def dry_run_demo():
    """模拟 API 调用的完整流程，理解数据是怎么流转的"""
    print("=== 模拟 API 调用流程（无需联网）===\n")

    # 这就是你发给 API 的数据结构
    request_data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个友好的助手。"},
            {"role": "user", "content": "用一句话介绍什么是 Python。"},
        ],
        "temperature": 0.7,
        "stream": False,
    }
    print("1. 发送给 API 的请求体：")
    for key, value in request_data.items():
        print(f"   {key}: {value}")

    # 这就是 API 返回的数据结构（实际调用时会收到真实内容）
    print("\n2. API 返回的响应结构：")
    print("""
    {
        "id": "chatcmpl-xxx",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Python 是一种简洁优雅的高级编程语言..."
            }
        }],
        "usage": {
            "prompt_tokens": 25,
            "completion_tokens": 15,
            "total_tokens": 40
        }
    }
    """)

    print("3. 代码中如何提取 AI 回复：")
    print('   reply = response.choices[0].message.content')
    print('   # reply 就是字符串: "Python 是一种简洁优雅的..."')
    print()
    print("这就是整个 API 调用的流程！只需要一个 HTTP POST 请求。")


if __name__ == "__main__":
    import sys

    # 如果没有 API key，自动进入学习模式
    if client.api_key == "your-deepseek-key":
        print("提示：未检测到 API Key，进入学习模式（演示数据结构，不联网）\n")
        dry_run_demo()

        print("=" * 50)
        print("如何获取真实的 API Key：")
        print("1. 访问 https://platform.deepseek.com 注册")
        print("2. 在 API Keys 页面创建 key（新用户有免费额度）")
        print("3. 在项目目录创建 .env 文件，写入：")
        print("   DEEPSEEK_API_KEY=sk-你的key")
        print("4. 重新运行本程序")
        print("=" * 50)
        sys.exit(0)

    print("=" * 50)
    print("演示1：简单对话")
    print("=" * 50)
    simple_chat()

    print("\n" + "=" * 50)
    print("演示2：角色扮演（system prompt）")
    print("=" * 50)
    explain_roles()

    print("\n" + "=" * 50)
    print("演示3：流式输出")
    print("=" * 50)
    streaming_chat()
