"""
LLM API 客户端 — 多平台适配 + 流式输出。

=== 核心概念 ===

1. OpenAI 兼容协议：
   几乎所有大模型 API（DeepSeek、通义千问、Ollama 等）都兼容 OpenAI 的接口格式。
   这意味着你只需要改 base_url 和 api_key，代码其他部分完全不变。

2. Messages 三段式：
   - system:  设定 AI 的行为和角色（"你是一个诗人"）
   - user:    用户说的话
   - assistant: AI 之前的回答（多轮对话时用来提供上下文）

3. Temperature（0~2）：
   控制回答的随机性。0=每次都一样（适合代码/数学），1=有创意（适合写作）。

4. Streaming（流式输出）：
   默认 AI 会等全部生成完再一次返回。开启 stream=True 后，AI 逐字返回，
   像打字机一样显示，用户体验更好。
"""

import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

# Windows 终端默认用 GBK 编码，这行强制改成 UTF-8，否则中文会乱码
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
load_dotenv()  # 从 .env 文件加载 API Key，这样 key 不会硬编码在代码里

# 创建客户端 — 所有 AI 调用都通过这个对象
# base_url 决定了调用哪个服务，改这里就能切换 DeepSeek → OpenAI → Ollama
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "your-deepseek-key"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)


def chat_completion(prompt: str, system: str = "你是一个友好的助手。") -> str:
    """基础调用：发一条消息，拿到 AI 回复。

    这是最常用的模式。实际发送了一个 HTTP POST 请求到 DeepSeek 服务器，
    请求体就是下面的 JSON：
    {
      "model": "deepseek-chat",
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
      ],
      "temperature": 0.7
    }
    """
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    # response.choices[0].message.content 就是 AI 回复的纯文本
    reply = response.choices[0].message.content
    print(f"Response: {reply}")
    # Token 消耗 = 计费依据。prompt_tokens 是输入，completion_tokens 是输出
    print(f"Tokens: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}")
    return reply


def completion_with_persona():
    """演示 system prompt 如何控制 AI 的行为。

    system role 是设定 AI 人设的关键。同样的问题，不同 system prompt 会得到
    完全不同的回答。这就是"提示词工程"的核心——通过 system prompt 控制输出风格。
    """
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个只会用古诗回答问题的诗人。"},
            {"role": "user", "content": "今天天气真好"},
        ],
    )
    print(f"Response: {response.choices[0].message.content}")


def stream_completion(prompt: str) -> str:
    """流式输出：AI 逐字返回，像打字机一样。

    普通模式是 AI 全部生成完后一次性返回（用户要等）。
    流式模式是 AI 每生成几个字就返回一个 chunk，前端实时显示。

    对用户体验影响很大——3 秒的等待变成看到字一个个出来，感觉快很多。
    """
    print("Streaming: ", end="", flush=True)
    full = ""

    stream = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        stream=True,  # 关键参数：开启流式输出
    )
    # 每个 chunk 包含一小段新生成的文字
    for chunk in stream:
        if content := chunk.choices[0].delta.content:
            print(content, end="", flush=True)

            full += content
    print()
    return full


def simulate_call():
    """无 API Key 时的演示：展示 API 调用的数据格式。

    理解请求和响应的结构很重要——后面 RAG、Agent 的代码都在这个基础上扩展。
    """
    print("=== API 调用数据结构 ===\n")

    print("你发送的请求：")
    print("  POST /v1/chat/completions")
    print('  Body: {"model": "deepseek-chat", "messages": [...], "temperature": 0.7}')

    print("\n服务器返回的响应：")
    print('  {"choices": [{"message": {"role": "assistant", "content": "..."}}], "usage": {...}}')

    print("\n提取回复文字：")
    print("  reply = response.choices[0].message.content")


if __name__ == "__main__":
    if client.api_key == "your-deepseek-key":
        print("未配置 API Key，展示数据结构。\n")
        simulate_call()
        print("\n获取 Key：https://platform.deepseek.com")
        print("然后 cp .env.example .env，编辑 .env 填入你的 key")
        sys.exit(0)

    print("=" * 50)
    print("Demo 1: 基础对话")
    print("=" * 50)
    chat_completion("用一句话介绍什么是 Python。")

    print("\n" + "=" * 50)
    print("Demo 2: system prompt 控制人设")
    print("=" * 50)
    completion_with_persona()

    print("\n" + "=" * 50)
    print("Demo 3: 流式输出")
    print("=" * 50)
    stream_completion("写一首关于编程的五言诗")
