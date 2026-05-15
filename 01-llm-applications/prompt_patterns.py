"""Prompt pattern library — reusable templates for common LLM tasks."""

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

MODEL = "deepseek-chat"

# ---------------------------------------------------------------
# Pattern 1: Role-based prompt
# ---------------------------------------------------------------
ROLE_TEMPLATE = """你是一个 {role}，擅长 {skill}。

任务：{task}

要求：
{requirements}"""


def demo_role_pattern():
    """Code review with role-based system prompt."""
    prompt = ROLE_TEMPLATE.format(
        role="Python 代码审查专家",
        skill="发现代码中的安全漏洞、性能问题和不良实践",
        task="审查以下代码并给出改进建议",
        requirements="1. 按严重程度排列\n2. 每个问题给出修复代码\n3. 用中文回答",
    )
    code = """def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    result = db.execute(query)
    return result"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": code},
        ],
    )
    print(response.choices[0].message.content)


# ---------------------------------------------------------------
# Pattern 2: Chain-of-thought
# ---------------------------------------------------------------
COT_TEMPLATE = """请一步步思考解决以下问题。按以下步骤回答：

1. 分析：理解问题的关键点
2. 方案：提出解决思路
3. 实现：给出具体代码/方案

问题：{problem}"""


def demo_cot_pattern():
    """Multi-step reasoning with chain-of-thought."""
    prompt = COT_TEMPLATE.format(
        problem="设计一个 Python 函数，输入一个字符串，判断它是否是有效的 IPv4 地址。"
    )
    response = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}]
    )
    print(response.choices[0].message.content)


# ---------------------------------------------------------------
# Pattern 3: Structured JSON output
# ---------------------------------------------------------------
STRUCTURED_TEMPLATE = """请严格按以下 JSON 格式回答，不要输出其他内容：

{format_spec}

现在请处理：{input_text}"""


def demo_structured_pattern():
    """Forcing structured JSON output for programmatic consumption."""
    prompt = STRUCTURED_TEMPLATE.format(
        format_spec='''{
    "sentiment": "正面/负面/中性",
    "keywords": ["关键词1", "关键词2"],
    "summary": "一句话总结"
}''',
        input_text="昨天新买的笔记本电脑屏幕碎了，但客服态度很好，直接给我换了一台新的。",
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # Low temperature for format consistency
    )
    print(response.choices[0].message.content)


# ---------------------------------------------------------------
# Pattern 4: Multi-turn conversation
# ---------------------------------------------------------------
def demo_multiturn():
    """Stateful multi-turn conversation with context persistence."""
    messages = [{"role": "system", "content": "你是一个乐于助人的助手。"}]
    print("Multi-turn chat (type quit to exit):")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "quit":
            break
        messages.append({"role": "user", "content": user_input})
        response = client.chat.completions.create(model=MODEL, messages=messages)
        reply = response.choices[0].message.content
        print(f"AI: {reply}")
        messages.append({"role": "assistant", "content": reply})


# ---------------------------------------------------------------
# Extra: Temperature tuning
# ---------------------------------------------------------------
def demo_temperature():
    """Compare outputs across different temperature values."""
    for temp in [0.1, 0.7, 1.5]:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "用一句话介绍什么是递归"}],
            temperature=temp,
        )
        print(f"t={temp}: {response.choices[0].message.content}")


if __name__ == "__main__":
    print("Pattern 1: Role-based (Code Review)")
    print("=" * 50)
    demo_role_pattern()

    print("\n\nPattern 2: Chain-of-Thought (IPv4 Validator)")
    print("=" * 50)
    demo_cot_pattern()

    print("\n\nPattern 3: Structured Output (Sentiment Analysis)")
    print("=" * 50)
    demo_structured_pattern()

    print("\n\nTemperature Comparison")
    print("=" * 50)
    demo_temperature()

    # Uncomment for interactive multi-turn:
    # demo_multiturn()
