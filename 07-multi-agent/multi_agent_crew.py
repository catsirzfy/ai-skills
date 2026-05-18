"""
CrewAI 风格多 Agent 协作 — 角色扮演 + 任务分配。

=== CrewAI 的核心思想 ===

把 Agent 组织成"团队"：
- 每个 Agent 有明确的角色（Role）、目标（Goal）、背景故事（Backstory）
- 任务（Task）分配给特定 Agent，有明确的期望输出
- 多个 Agent 按顺序或并行执行任务

=== 和 LangGraph 多 Agent 的区别 ===

CrewAI：高层抽象，定义角色 → 自动分配任务 → 执行
LangGraph：底层控制，手动定义节点 → 边 → 条件路由

=== 本文件演示 ===

一个代码审查团队：
- 架构师 Agent：审查代码架构设计
- 安全专家 Agent：审查安全漏洞
- 性能专家 Agent：审查性能问题
- 项目经理 Agent：汇总审查报告
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "your-deepseek-key"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)
MODEL = "deepseek-chat"


# ================================================================
# Agent 定义（CrewAI 风格：角色 + 目标 + 背景）
# ================================================================
AGENTS = {
    "架构师": {
        "role": "软件架构师",
        "goal": "审查代码的架构设计、模块划分、设计模式使用",
        "backstory": "10年经验的资深架构师，擅长系统设计和代码组织。",
    },
    "安全专家": {
        "role": "安全专家",
        "goal": "审查代码的安全漏洞：SQL注入、XSS、权限问题、敏感信息泄露",
        "backstory": "信息安全专家，曾发现多个知名项目的安全漏洞。",
    },
    "性能专家": {
        "role": "性能专家",
        "goal": "审查代码的性能问题：时间复杂度、不必要的计算、内存使用",
        "backstory": "性能优化专家，专注于高并发系统的性能优化。",
    },
}


def run_agent(name: str, config: dict, task: str) -> str:
    """运行单个 Agent。"""
    print(f"\n  [{name}] 开始审查...")

    # 用 CrewAI 的格式构建 prompt
    system_prompt = (
        f"你是{config['role']}。\n"
        f"背景：{config['backstory']}\n"
        f"审查目标：{config['goal']}\n"
        f"规则：\n"
        f"1. 只关注你的专业领域，不要跨领域评价\n"
        f"2. 每个问题按严重程度排序（严重/中等/建议）\n"
        f"3. 给出具体的代码位置和修复建议\n"
        f"4. 如果该领域没有问题，明确说明'未发现相关问题'\n"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请审查以下代码：\n```python\n{task}\n```"},
        ],
    )
    result = response.choices[0].message.content
    print(f"  [{name}] 完成（{len(result)} 字）")
    return result


def run_crew_review(code: str):
    """运行完整的代码审查团队。

    流程：
    1. 并行：架构师、安全专家、性能专家同时审查
    2. 串行：项目经理汇总三份报告 → 生成最终审查报告
    """
    print("=" * 60)
    print("代码审查团队")
    print("=" * 60)
    print(f"待审查代码:\n```python\n{code}\n```\n")

    # --- 阶段 1：并行审查 ---
    print("阶段 1：并行审查（架构 + 安全 + 性能）")
    results = {}
    for name, config in AGENTS.items():
        results[name] = run_agent(name, config, code)

    # --- 阶段 2：汇总报告 ---
    print(f"\n[项目经理] 正在汇总审查报告...")

    summary_prompt = f"""你是项目经理。请将以下三位专家的审查报告汇总成一份完整的代码审查报告。

报告格式：
1. 总体评价（1-10 分）
2. 严重问题（必须修复）
3. 中等问题（建议修复）
4. 改进建议
5. 总结

---
架构师报告：
{results['架构师']}

安全专家报告：
{results['安全专家']}

性能专家报告：
{results['性能专家']}
---"""

    summary = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "你是项目经理，负责汇总技术审查报告。使用中文，清晰分层。"},
            {"role": "user", "content": summary_prompt},
        ],
    ).choices[0].message.content

    print("=" * 60)
    print("最终审查报告")
    print("=" * 60)
    print(summary)

    return {**results, "summary": summary}


if __name__ == "__main__":
    # 故意包含多种问题的示例代码
    sample_code = """
import sqlite3

class UserService:
    def __init__(self):
        self.db = sqlite3.connect("users.db")

    def get_user(self, user_id):
        query = "SELECT * FROM users WHERE id = " + str(user_id)
        return self.db.execute(query).fetchone()

    def get_all_users(self):
        result = self.db.execute("SELECT * FROM users")
        users = []
        for row in result:
            users.append(row)
        return users

    def save_user(self, name, email):
        for i in range(len(name)):
            if name[i] == "'":
                name = name[:i] + "''" + name[i+1:]
        query = f"INSERT INTO users (name, email) VALUES ('{name}', '{email}')"
        self.db.execute(query)
        self.db.commit()
"""

    run_crew_review(sample_code.strip())
