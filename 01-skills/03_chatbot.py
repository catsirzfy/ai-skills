"""
第3课：交互式聊天机器人
目标：整合前面学的知识，做一个能用的命令行聊天机器人

功能：
- 多轮对话记忆
- 流式输出
- /save 保存对话历史
- /clear 清除记忆
- /system 修改系统提示词
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


class ChatBot:
    """一个简单的 AI 聊天机器人"""

    def __init__(self, system_prompt="你是一个友好的 AI 助手。"):
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]

    def chat(self, user_input: str) -> str:
        """发送消息并获取回复"""
        self.messages.append({"role": "user", "content": user_input})

        # 流式输出
        print("AI: ", end="", flush=True)
        full_reply = ""

        stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=self.messages,
            stream=True,
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
                full_reply += content

        print()
        self.messages.append({"role": "assistant", "content": full_reply})
        return full_reply

    def clear_history(self):
        """清空对话历史，但保留 system prompt"""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def set_system_prompt(self, prompt: str):
        """修改系统提示词"""
        self.system_prompt = prompt
        self.messages[0] = {"role": "system", "content": prompt}

    def save_history(self, filename: str = None):
        """保存对话历史到文件"""
        if not filename:
            filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        save_path = os.path.join(os.path.dirname(__file__), filename)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)
        print(f"对话已保存到: {save_path}")

    def load_history(self, filename: str):
        """从文件加载对话历史"""
        load_path = os.path.join(os.path.dirname(__file__), filename)
        with open(load_path, "r", encoding="utf-8") as f:
            self.messages = json.load(f)
        self.system_prompt = self.messages[0]["content"]
        print(f"已加载对话: {filename} (共 {len(self.messages)} 条消息)")


def main():
    print("=" * 50)
    print("AI 聊天机器人")
    print("=" * 50)
    print("命令:")
    print("  /clear   - 清空对话记忆")
    print("  /system  - 修改 AI 角色（例：/system 你是诗人）")
    print("  /save    - 保存对话历史")
    print("  /load    - 加载对话历史（例：/load chat.json）")
    print("  /quit    - 退出")
    print("=" * 50)

    bot = ChatBot()

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # 处理命令
        if user_input.startswith("/"):
            cmd, *args = user_input.split(maxsplit=1)
            arg = args[0] if args else ""

            if cmd == "/quit":
                print("再见！")
                break
            elif cmd == "/clear":
                bot.clear_history()
                print("对话记忆已清空。")
            elif cmd == "/system":
                new_prompt = arg or "你是一个友好的 AI 助手。"
                bot.set_system_prompt(new_prompt)
                print(f"系统提示词已更新为: {new_prompt}")
            elif cmd == "/save":
                bot.save_history(arg if arg else None)
            elif cmd == "/load":
                if arg:
                    bot.load_history(arg)
                else:
                    print("请指定文件名，例：/load chat.json")
            else:
                print(f"未知命令: {cmd}")
            continue

        # 正常对话
        bot.chat(user_input)


if __name__ == "__main__":
    main()
