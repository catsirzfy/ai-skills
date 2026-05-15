"""
第2课：提示词工程 (Prompt Engineering)
目标：掌握 4 种实用的提示词模板，能在任何项目中复用

核心原则：把话说清楚 = 角色 + 任务 + 约束
"""

import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

# Windows 终端 UTF-8 编码修复
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "your-deepseek-key"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

# ============================================================
# 模板1：角色扮演（最常用，80% 的场景用这个）
# ============================================================
TEMPLATE_ROLE = """
你是一个 {role}，擅长 {skill}。

任务：{task}

要求：
{requirements}
"""


def role_play_example():
    """演示角色扮演模板"""
    prompt = TEMPLATE_ROLE.format(
        role="Python 代码审查专家",
        skill="发现代码中的安全漏洞、性能问题和不良实践",
        task="审查以下代码并给出改进建议",
        requirements="1. 按严重程度排列问题\n2. 每个问题给出具体修复代码\n3. 用中文回答",
    )

    code_to_review = """
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    result = db.execute(query)
    return result
    """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": code_to_review},
        ],
    )
    print(response.choices[0].message.content)


# ============================================================
# 模板2：链式思考（复杂推理/逻辑问题用）
# ============================================================
TEMPLATE_COT = """
请一步步思考解决以下问题。按以下步骤回答：

1. 分析：理解问题的关键点
2. 方案：提出解决思路
3. 实现：给出具体代码/方案

问题：{problem}
"""


def chain_of_thought_example():
    """演示链式思考模板"""
    prompt = TEMPLATE_COT.format(
        problem="设计一个 Python 函数，输入一个字符串，判断它是否是有效的 IPv4 地址。"
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
    )
    print(response.choices[0].message.content)


# ============================================================
# 模板3：结构化输出（需要解析结果时用）
# ============================================================
TEMPLATE_STRUCTURED = """
请严格按以下 JSON 格式回答，不要输出其他内容：

{format_spec}

现在请处理：{input_text}
"""


def structured_output_example():
    """演示结构化输出模板 - 适合程序化处理 AI 结果"""
    prompt = TEMPLATE_STRUCTURED.format(
        format_spec='''
{
    "sentiment": "正面/负面/中性",
    "keywords": ["关键词1", "关键词2"],
    "summary": "一句话总结"
}
''',
        input_text="昨天新买的笔记本电脑屏幕碎了，但客服态度很好，直接给我换了一台新的。",
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # 结构化输出用低温，保证格式稳定
    )
    print(response.choices[0].message.content)


# ============================================================
# 模板4：多轮对话记忆
# ============================================================
def multi_turn_conversation():
    """演示多轮对话 - AI 能记住前面说过的话"""
    messages = [
        {"role": "system", "content": "你是一个乐于助人的助手。"},
    ]

    print("多轮对话演示（输入 quit 退出）：")
    while True:
        user_input = input("\n你说: ")
        if user_input.lower() == "quit":
            break

        # 把用户消息加入对话历史
        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
        )

        ai_reply = response.choices[0].message.content
        print(f"AI: {ai_reply}")

        # 把 AI 回复也加入对话历史（这样 AI 就记得之前说过什么）
        messages.append({"role": "assistant", "content": ai_reply})


# ============================================================
# 额外技巧：temperature 和 top_p 的区别
# ============================================================
def temperature_demo():
    """对比不同 temperature 的效果"""
    question = "用一句话介绍什么是递归"

    for temp in [0.1, 0.7, 1.5]:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": question}],
            temperature=temp,
        )
        print(f"\ntemperature={temp}: {response.choices[0].message.content}")


if __name__ == "__main__":
    print("模板1：角色扮演（代码审查）")
    print("=" * 50)
    role_play_example()

    print("\n\n模板2：链式思考（IPv4 验证）")
    print("=" * 50)
    chain_of_thought_example()

    print("\n\n模板3：结构化输出（情感分析）")
    print("=" * 50)
    structured_output_example()

    print("\n\nTemperature 对比")
    print("=" * 50)
    temperature_demo()

    # 取消注释来体验多轮对话：
    # multi_turn_conversation()
