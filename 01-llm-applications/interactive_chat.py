"""
流式对话引擎 — 会话持久化 + 动态人设切换。

=== 核心概念 ===

1. 对话状态管理：
   AI 本身没有记忆。所有"记忆"都在 messages 列表里。
   每发一条消息，把整个 messages 列表传给 AI，再把 AI 的回复加回列表。

2. 流式输出：
   普通调用：AI 生成完所有文字才返回 → 用户等待时间长
   流式调用：AI 每生成几个字就返回 → 用户实时看到文字，体验好

3. 会话持久化：
   messages 保存在 JSON 文件里，下次打开可以继续之前的话题。

4. 动态人设：
   运行时修改 system prompt，让同一个对话窗口能切换不同角色。
"""

import os
import sys
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "your-deepseek-key"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

MODEL = "deepseek-chat"
HISTORY_DIR = os.path.dirname(__file__)


class ChatSession:
    """管理对话状态：消息历史、人设、持久化。

    这是 AI 聊天应用的标准模式。几乎所有聊天产品（ChatGPT、Kimi 等）
    底层都是类似的 messages 列表管理。
    """

    def __init__(self, system_prompt: str = "你是一个友好的 AI 助手。"):
        self.system_prompt = system_prompt
        # messages 是核心数据结构：一个 dict 列表，记录全部对话
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]

    def send(self, text: str) -> str:
        """发送消息并通过流式输出获取回复。

        流程：
        1. 把用户消息加入 messages
        2. 发请求给 AI（stream=True 逐字接收）
        3. 把 AI 回复也加入 messages（维持上下文）
        """
        self.messages.append({"role": "user", "content": text})
        print("AI: ", end="", flush=True)
        full = ""

        stream = client.chat.completions.create(
            model=MODEL, messages=self.messages, stream=True
        )
        for chunk in stream:
            # delta.content 是新生成的文字片段（不是完整消息）
            if content := chunk.choices[0].delta.content:
                print(content, end="", flush=True)
                full += content
        print()
        # 关键：把 AI 的回复加回 messages，下次才能继续上下文
        self.messages.append({"role": "assistant", "content": full})
        return full

    def set_persona(self, prompt: str):
        """运行时修改 AI 人设。

        直接替换 messages[0] 的 system prompt，
        不需要重新创建 session。
        """
        self.system_prompt = prompt
        self.messages[0] = {"role": "system", "content": prompt}

    def reset(self):
        """清空对话，但保留人设。

        只删除对话历史（user 和 assistant 消息），
        system prompt 保持不变。
        """
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def save(self, filename: str | None = None):
        """保存整个 messages 列表到 JSON 文件。

        保存的 JSON 就是 messages 的完整快照，
        下次加载后可以无缝继续对话。
        """
        path = filename or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        full_path = os.path.join(HISTORY_DIR, path)
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)
        print(f"Saved: {full_path}")

    def load(self, filename: str):
        """从 JSON 文件恢复对话。

        加载后 messages 列表完全恢复，可以接着之前的话题继续聊。
        """
        path = os.path.join(HISTORY_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            self.messages = json.load(f)
        self.system_prompt = self.messages[0]["content"]
        print(f"Loaded: {filename} ({len(self.messages)} messages)")


def run():
    """启动交互式对话终端。"""
    session = ChatSession()
    print("=" * 50)
    print("流式对话引擎")
    print("=" * 50)
    print("命令：")
    print("  /clear   — 清空对话记忆")
    print("  /system  — 切换 AI 人设（例：/system 你是一个诗人）")
    print("  /save    — 保存当前对话到文件")
    print("  /load    — 从文件恢复对话")
    print("  /quit    — 退出")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd, _, arg = user_input.partition(" ")

            if cmd == "/quit":
                print("Bye.")
                break
            elif cmd == "/clear":
                session.reset()
                print("对话记忆已清空。")
            elif cmd == "/system":
                session.set_persona(arg or "你是一个友好的 AI 助手。")
                print(f"人设已更新: {session.system_prompt}")
            elif cmd == "/save":
                session.save(arg or None)
            elif cmd == "/load":
                if arg:
                    session.load(arg)
                else:
                    print("用法: /load <文件名>")
            else:
                print(f"未知命令: {cmd}")
        else:
            session.send(user_input)


if __name__ == "__main__":
    run()
