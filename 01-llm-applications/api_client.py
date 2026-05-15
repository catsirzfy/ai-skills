"""LLM API client — multi-provider adapter with streaming support.

Works with any OpenAI-compatible endpoint (DeepSeek, Qwen, Ollama, etc.).
"""

import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "your-deepseek-key"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)


def chat_completion(prompt: str, system: str = "你是一个友好的助手。") -> str:
    """Single-turn completion with system prompt."""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    print(f"Response: {response.choices[0].message.content}")
    print(f"Tokens: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}")
    return response.choices[0].message.content


def completion_with_persona():
    """Demonstrate system-prompt-driven persona control."""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个只会用古诗回答问题的诗人。"},
            {"role": "user", "content": "今天天气真好"},
        ],
    )
    print(f"Response: {response.choices[0].message.content}")


def stream_completion(prompt: str) -> str:
    """Completion with real-time streaming output."""
    print("Streaming: ", end="", flush=True)
    full = ""

    stream = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        if content := chunk.choices[0].delta.content:
            print(content, end="", flush=True)
            full += content
    print()
    return full


def simulate_call():
    """Show the request/response structure without making an actual API call."""
    print("=== API Call Structure (dry-run) ===\n")

    print("Request payload:")
    print("  POST /v1/chat/completions")
    print('  {"model": "deepseek-chat", "messages": [...], "temperature": 0.7}')

    print("\nResponse payload:")
    print('  {"choices": [{"message": {"role": "assistant", "content": "..."}}], "usage": {...}}')

    print("\nExtract reply: response.choices[0].message.content")


if __name__ == "__main__":
    if client.api_key == "your-deepseek-key":
        print("No API key configured. Showing dry-run mode.\n")
        simulate_call()

        print("\n" + "=" * 50)
        print("To configure:")
        print("  1. Register at https://platform.deepseek.com")
        print("  2. cp .env.example .env")
        print("  3. Edit .env with your key")
        print("=" * 50)
        sys.exit(0)

    print("=" * 50)
    print("Demo 1: Single-turn completion")
    print("=" * 50)
    chat_completion("用一句话介绍什么是 Python。")

    print("\n" + "=" * 50)
    print("Demo 2: Persona injection via system prompt")
    print("=" * 50)
    completion_with_persona()

    print("\n" + "=" * 50)
    print("Demo 3: Streaming completion")
    print("=" * 50)
    stream_completion("写一首关于编程的五言诗")
