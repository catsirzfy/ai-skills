"""
Agent Skill 3：规划（Planning）

=== 为什么需要规划 ===

复杂任务无法一步完成。比如"帮我分析这个 CSV 文件并画出销售趋势图"：
- 需要读文件 → 理解数据 → 写分析代码 → 执行 → 画图 → 总结
- Agent 需要自己拆解任务，决定先做什么、后做什么

=== 三种规划方式 ===

1. ReAct（Reasoning + Acting）
   思考 → 行动 → 观察 → 再思考 → ... → 最终答案
   最经典的 Agent 模式，LangChain 和 LangGraph 都基于此。

2. Plan-and-Execute（先计划再执行）
   先列出完整计划 → 用户确认 → 逐步执行
   适合确定性高的任务。

3. 结构化分解
   把用户模糊的需求分解成具体的子任务。
"""

import os
import sys
import json
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
# 规划模式 1：ReAct（思考 → 行动 → 观察 循环）
# ================================================================
def react_agent(task: str, max_steps: int = 5) -> str:
    """ReAct 模式 Agent——思考和行动交替进行。

    每轮：
    Thought: 我现在应该做什么？
    Action:  调用什么工具？传什么参数？
    Observation: 工具返回了什么？
    然后循环，直到得出 Final Answer。

    这是 LangGraph Agent 的底层原理——先理解这个，LangGraph 就只是自动化了这个循环。
    """
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个能思考并行动的智能体。请按以下格式回答：\n\n"
                "Thought: 我的思考过程...\n"
                "Action: 工具名\n"
                "Action Input: 工具参数（JSON）\n"
                "...（重复 Thought/Action/Action Input，直到得出答案）...\n"
                "Final Answer: 最终答案\n\n"
                "可用工具：\n"
                "- search(query: str) — 搜索信息\n"
                "- calculate(expression: str) — 数学计算\n"
                "- read_file(path: str) — 读取文件（模拟）\n"
                "- done(result: str) — 任务完成\n\n"
                "重要规则：每轮必须有 Thought 和 Action，直到 can 调用 done 结束。"
            ),
        },
        {"role": "user", "content": f"任务：{task}"},
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0.3
        )
        text = response.choices[0].message.content
        print(f"\n--- Step {step + 1} ---")
        print(text)

        messages.append({"role": "assistant", "content": text})

        # 检查是否结束
        if "Final Answer:" in text:
            final = text.split("Final Answer:")[-1].strip()
            print(f"\n✅ 任务完成: {final}")
            return final

        # 模拟工具执行（简化版）
        if "Action:" in text and "Action Input:" in text:
            action_part = text.split("Action:")[-1].split("Action Input:")[0].strip()
            try:
                args_str = text.split("Action Input:")[-1].split("\n")[0].strip()
                args = json.loads(args_str) if args_str.startswith("{") else {"query": args_str}
            except (json.JSONDecodeError, IndexError):
                args = {"input": "unknown"}

            # 模拟搜索结果
            if "search" in action_part.lower():
                observation = f"搜索结果：{args} 是 AI 开发中的重要概念。"
            elif "calculate" in action_part.lower():
                try:
                    result = eval(args.get("expression", "0"), {"__builtins__": {}}, {})
                    observation = f"计算结果: {result}"
                except Exception:
                    observation = "计算错误"
            elif "done" in action_part.lower():
                print(f"\n✅ 完成: {args}")
                return str(args)
            else:
                observation = f"工具 {action_part} 执行成功。"

            print(f"  📋 观察结果: {observation}")
            messages.append({"role": "user", "content": f"Observation: {observation}"})

    return "达到最大步数限制。"


# ================================================================
# 规划模式 2：Plan-and-Execute（先计划再执行）
# ================================================================
def plan_then_execute(task: str) -> list[str]:
    """先让 LLM 拆解任务成步骤列表，再展示每个步骤。

    这适合给用户看的场景——先展示完整计划，用户确认后再执行。
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "system",
            "content": (
                "你是一个任务规划助手。收到任务后，把它拆解成 3-5 个具体的执行步骤。"
                "每个步骤一行，格式：'步骤N: 具体做什么'"
                "只输出步骤列表，不要解释。"
            ),
        },
        {"role": "user", "content": f"请为以下任务制定计划：{task}"}],
        temperature=0.3,
    )
    plan = response.choices[0].message.content.strip()
    steps = [s.strip() for s in plan.split("\n") if s.strip()]
    return steps


# ================================================================
# 规划模式 3：结构化任务分解
# ================================================================
def decompose_task(goal: str) -> dict:
    """把模糊目标分解成结构化子任务。

    返回：{"goal": ..., "subtasks": [{"title": ..., "tool": ..., "params": ...}, ...]}
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "system",
            "content": (
                "你是一个任务分解助手。把用户目标分解成 3-5 个子任务。"
                "输出 JSON 格式："
                '{"goal": "原目标", "subtasks": [{"title": "子任务名", "tool_needed": "所需工具", "description": "做什么"}]}'
            ),
        },
        {"role": "user", "content": f"目标：{goal}"}],
        temperature=0.1,
    )
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"goal": goal, "subtasks": [], "error": "解析失败"}


# ================================================================
# 演示
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("规划模式 1：ReAct 循环")
    print("=" * 60)
    react_agent("帮我算一下 (12345 * 67890) / 100 的结果，然后告诉我这个数字大不大")

    print("\n\n" + "=" * 60)
    print("规划模式 2：先计划再执行")
    print("=" * 60)
    steps = plan_then_execute("帮我写一个分析 CSV 文件并生成销售趋势图的 Python 脚本")
    for s in steps:
        print(f"  {s}")

    print("\n\n" + "=" * 60)
    print("规划模式 3：结构化任务分解")
    print("=" * 60)
    result = decompose_task("分析公司销售数据，找出趋势并生成报告")
    print(f"  目标: {result['goal']}")
    for st in result.get("subtasks", []):
        print(f"    子任务: {st['title']} → 需要: {st['tool_needed']}")
