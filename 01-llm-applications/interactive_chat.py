"""Streaming chat engine with session persistence and dynamic persona switching."""

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
    """Manages chat state: message history, persona, and persistence."""

    def __init__(self, system_prompt: str = "你是一个友好的 AI 助手。"):
        self.system_prompt = system_prompt
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]

    def send(self, text: str) -> str:
        """Send a message and yield the response via streaming."""
        self.messages.append({"role": "user", "content": text})
        print("AI: ", end="", flush=True)
        full = ""

        stream = client.chat.completions.create(
            model=MODEL, messages=self.messages, stream=True
        )
        for chunk in stream:
            if content := chunk.choices[0].delta.content:
                print(content, end="", flush=True)
                full += content
        print()
        self.messages.append({"role": "assistant", "content": full})
        return full

    def set_persona(self, prompt: str):
        """Switch the system prompt at runtime."""
        self.system_prompt = prompt
        self.messages[0] = {"role": "system", "content": prompt}

    def reset(self):
        """Clear conversation history, retaining the persona."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def save(self, filename: str | None = None):
        """Persist session to a JSON file."""
        path = filename or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        full_path = os.path.join(HISTORY_DIR, path)
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)
        print(f"Saved: {full_path}")

    def load(self, filename: str):
        """Restore a previously saved session."""
        path = os.path.join(HISTORY_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            self.messages = json.load(f)
        self.system_prompt = self.messages[0]["content"]
        print(f"Loaded: {filename} ({len(self.messages)} messages)")


def run():
    session = ChatSession()
    print("=" * 50)
    print("Streaming Chat Engine")
    print("=" * 50)
    print("Commands:")
    print("  /clear   — Reset conversation")
    print("  /system  — Change AI persona")
    print("  /save    — Persist session")
    print("  /load    — Restore session")
    print("  /quit    — Exit")
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
                print("Session reset.")
            elif cmd == "/system":
                session.set_persona(arg or "你是一个友好的 AI 助手。")
                print(f"Persona updated: {session.system_prompt}")
            elif cmd == "/save":
                session.save(arg or None)
            elif cmd == "/load":
                if arg:
                    session.load(arg)
                else:
                    print("Usage: /load <filename>")
            else:
                print(f"Unknown: {cmd}")
        else:
            session.send(user_input)


if __name__ == "__main__":
    run()
