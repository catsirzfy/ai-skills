"""
提示词模式库 — 4 种可复用的提示词模板。

=== 为什么需要提示词模板 ===

同样的 AI 模型，不同的提示词会得到完全不同的输出质量。
模板让你的提示词结构化和可复用，而不是每次都从头想。

=== 四种模板 ===

1. 角色扮演（Role-based）— 80% 场景用这个
   给 AI 设定专业角色 + 任务 + 要求，让它在特定领域内回答。

2. 链式思考（Chain-of-Thought）— 复杂推理
   强制 AI 一步步思考，而不是直接给答案。对数学/逻辑问题效果显著。

3. 结构化输出（Structured Output）— 需要程序解析结果
   要求 AI 输出 JSON，方便后续代码自动处理。

4. 多轮对话（Multi-turn）— 上下文记忆
   把 messages 列表持续追加，AI 就能记住之前聊过的内容。
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

MODEL = "deepseek-chat"


# ================================================================
# 模式一：角色扮演 — 最常用的提示词模式
# ================================================================
# 核心技巧：system prompt 里明确角色 + 技能 + 任务 + 要求
# 这比单纯在 user 消息里写指令效果好得多
ROLE_TEMPLATE = """你是一个 {role}，擅长 {skill}。

任务：{task}

要求：
{requirements}"""


def demo_role_pattern():
    """用角色扮演模板做代码审查。

    system prompt 让 AI 代入"代码审查专家"的身份，
    它的回答会更专业、更有针对性。
    """
    prompt = ROLE_TEMPLATE.format(
        role="Python 代码审查专家",
        skill="发现代码中的安全漏洞、性能问题和不良实践",
        task="审查以下代码并给出改进建议",
        requirements="1. 按严重程度排列\n2. 每个问题给出修复代码\n3. 用中文回答",
    )
    # 故意写了一段有 SQL 注入漏洞的代码让 AI 审查
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


# ================================================================
# 模式二：链式思考 — 复杂问题的推理利器
# ================================================================
# 核心技巧：明确要求 AI 按 "分析 → 方案 → 实现" 的步骤回答
# 不写这个模板，AI 可能直接给代码，跳过分析过程
COT_TEMPLATE = """请一步步思考解决以下问题。按以下步骤回答：

1. 分析：理解问题的关键点
2. 方案：提出解决思路
3. 实现：给出具体代码/方案

问题：{problem}"""


def demo_cot_pattern():
    """用链式思考设计 IPv4 验证函数。

    如果不加 COT，AI 可能直接丢出一个正则表达式。
    加了 COT 后，AI 会先分析 IPv4 规则，再设计算法，最后才写代码。
    """
    prompt = COT_TEMPLATE.format(
        problem="设计一个 Python 函数，输入一个字符串，判断它是否是有效的 IPv4 地址。"
    )
    response = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}]
    )
    print(response.choices[0].message.content)


# ================================================================
# 模式三：结构化输出 — 让 AI 返回 JSON，方便程序解析
# ================================================================
# 核心技巧：
#   1. 明确要求 JSON 格式
#   2. 给出 JSON schema 示例
#   3. temperature 设低（0.1），减少格式错误
STRUCTURED_TEMPLATE = """请严格按以下 JSON 格式回答，不要输出其他内容：

{format_spec}

现在请处理：{input_text}"""


def demo_structured_pattern():
    """让 AI 做情感分析并返回结构化 JSON。

    这种模式在实际项目中非常实用——AI 分析文本，返回 JSON，
    你的代码直接 json.loads() 解析，不用手动从自然语言里提取信息。
    """
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
        temperature=0.1,  # 低温 = 输出更稳定，JSON 格式更可靠
    )
    print(response.choices[0].message.content)


# ================================================================
# 模式四：多轮对话 — AI 记住上下文
# ================================================================
# 核心技巧：messages 列表持续追加，不要每次都发送新的 messages
# AI 本身是无状态的，所有"记忆"都是你把历史消息反复传给它
def demo_multiturn():
    """演示多轮对话记忆机制。

    关键点：把 AI 的每次回复也 append 回 messages 列表，
    这样下一轮 AI 就知道之前聊了什么。
    如果你不 append assistant 的回复，AI 看不到自己说过的话。

    注意：每轮都要把整个 messages 列表发过去，
    Token 消耗会随对话增长，长对话需要注意截断。
    """
    messages = [{"role": "system", "content": "你是一个乐于助人的助手。"}]
    print("多轮对话（输入 quit 退出）：")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "quit":
            break
        messages.append({"role": "user", "content": user_input})
        response = client.chat.completions.create(model=MODEL, messages=messages)
        reply = response.choices[0].message.content
        print(f"AI: {reply}")
        # 关键：把 AI 的回复也加回 messages，否则下一轮 AI 不知道它说过什么
        messages.append({"role": "assistant", "content": reply})


# ================================================================
# 附：Temperature 对比
# ================================================================
def demo_temperature():
    """直观对比不同 temperature 的输出差异。

    t=0.1：几乎每次都一样（适合事实性问答）
    t=0.7：有一定变化但不离谱（适合日常对话）
    t=1.5：天马行空，每次都可能不一样（适合创意写作）
    """
    for temp in [0.1, 0.7, 1.5]:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "用一句话介绍什么是递归"}],
            temperature=temp,
        )
        print(f"t={temp}: {response.choices[0].message.content}")


if __name__ == "__main__":
    print("模式1：角色扮演（代码审查）")
    print("=" * 50)
    demo_role_pattern()

    print("\n\n模式2：链式思考（IPv4 验证器设计）")
    print("=" * 50)
    demo_cot_pattern()

    print("\n\n模式3：结构化输出（情感分析 → JSON）")
    print("=" * 50)
    demo_structured_pattern()

    print("\n\n温度参数对比")
    print("=" * 50)
    demo_temperature()

    # 取消注释下面这行来体验多轮对话：
    # demo_multiturn()
