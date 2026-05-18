"""
Agent Skill 2：记忆（Memory）

=== 为什么需要记忆 ===

LLM 每次调用都是"失忆"的。你发的每句话都是一个独立的 HTTP 请求。
要让 AI 记住上下文，你必须手动管理 messages 列表。

Agent 的 Memory 就是"自动管理 messages 列表的系统"。

=== 三种记忆类型 ===

1. Buffer（缓冲区记忆）
   最简单：把最近 N 轮对话存在 messages 里。
   优点：简单直接。缺点：对话长了 Token 爆炸。

2. Summary（摘要记忆）
   用 LLM 把长对话总结成一段摘要，只保留摘要。
   优点：省 Token。缺点：可能丢失细节。

3. ConversationBufferWindow（滑动窗口）
   只保留最近 K 轮对话，旧对话直接丢弃。
   优点：简单 + Token 可控。缺点：丢失早期上下文。
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
# 记忆类型 1：Buffer Memory（全量缓冲区）
# ================================================================
class BufferMemory:
    """保存完整的对话历史。最简单但 Token 消耗随对话增长。"""

    def __init__(self, system_prompt: str = "你是一个友好的助手。"):
        self.messages = [{"role": "system", "content": system_prompt}]

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text})

    def add_ai(self, text: str):
        self.messages.append({"role": "assistant", "content": text})

    def chat(self, user_input: str) -> str:
        self.add_user(user_input)
        response = client.chat.completions.create(model=MODEL, messages=self.messages)
        reply = response.choices[0].message.content
        self.add_ai(reply)
        return reply

    def token_estimate(self) -> int:
        """估算当前已用的 Token 数（中文字符粗略估计）。"""
        return sum(len(m["content"]) for m in self.messages) // 2


# ================================================================
# 记忆类型 2：Sliding Window Memory（滑动窗口）
# ================================================================
class WindowMemory:
    """只保留最近 K 轮对话。简单可控，但丢失早期上下文。"""

    def __init__(self, system_prompt: str = "你是一个友好的助手。", k: int = 3):
        self.system_prompt = system_prompt
        self.k = k
        self.history: list[tuple[str, str]] = []  # [(user_msg, ai_msg), ...]

    def chat(self, user_input: str) -> str:
        # 构建 messages：system + 最近 k 轮
        messages = [{"role": "system", "content": self.system_prompt}]
        for u, a in self.history[-self.k:]:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(model=MODEL, messages=messages)
        reply = response.choices[0].message.content
        self.history.append((user_input, reply))
        return reply


# ================================================================
# 记忆类型 3：Summary Memory（摘要记忆）
# ================================================================
class SummaryMemory:
    """用 LLM 自动生成对话摘要。省 Token，适合长对话。"""

    def __init__(self, system_prompt: str = "你是一个友好的助手。"):
        self.system_prompt = system_prompt
        self.summary = ""  # 累积的对话摘要
        self.recent: list[tuple[str, str]] = []  # 最近几轮（还没被摘要的）

    def _summarize(self):
        """用 LLM 把最近对话总结成摘要。"""
        if len(self.recent) < 3:
            return

        dialogue = "\n".join(f"用户: {u}\n助手: {a}" for u, a in self.recent)
        # 如果已有摘要，和新对话合并
        existing = f"\n之前的摘要: {self.summary}" if self.summary else ""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": f"请将以下对话总结为一段简洁的摘要，保留关键信息：{existing}\n\n对话：\n{dialogue}",
            }],
        )
        self.summary = response.choices[0].message.content
        self.recent = []  # 清空，新的对话从头累积

    def chat(self, user_input: str) -> str:
        if len(self.recent) >= 3:
            self._summarize()

        # 构建 messages：system + 摘要 + 最近对话
        messages = [{"role": "system", "content": self.system_prompt}]
        if self.summary:
            messages.append({"role": "system", "content": f"[对话摘要] {self.summary}"})

        for u, a in self.recent:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(model=MODEL, messages=messages)
        reply = response.choices[0].message.content
        self.recent.append((user_input, reply))
        return reply


# ================================================================
# 演示
# ================================================================
def demo():
    print("=" * 60)
    print("演示：三种记忆对比 — 同一段对话")
    print("=" * 60)

    conversation = [
        "我叫张三，是一个 Python 后端开发。",
        "我最近在学 AI 应用开发。",
        "我刚才说我叫什么来着？",
        "那根据我的背景，你觉得我应该学什么？",
    ]

    print("\n--- Buffer Memory（全量，记住一切）---")
    mem1 = BufferMemory()
    for q in conversation:
        reply = mem1.chat(q)
        print(f"  Q: {q}")
        print(f"  A: {reply}")
    print(f"  Token 估算: {mem1.token_estimate()}")

    print("\n--- Window Memory（k=2，只记最近 2 轮）---")
    mem2 = WindowMemory(k=2)
    for q in conversation:
        reply = mem2.chat(q)
        print(f"  Q: {q}")
        print(f"  A: {reply}")

    print("\n--- Summary Memory（自动摘要）---")
    mem3 = SummaryMemory()
    for q in conversation:
        reply = mem3.chat(q)
        print(f"  Q: {q}")
        print(f"  A: {reply}")
    print(f"  自动摘要内容: {mem3.summary[:150]}...")


if __name__ == "__main__":
    demo()
