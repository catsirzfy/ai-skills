"""
LangGraph 多 Agent 协作 — 两个 Agent 分工合作。

=== 多 Agent 核心概念 ===

单个 Agent 什么都能干，但能力分散。多 Agent 让每个 Agent 专注一个领域：
- 研究员 Agent：搜索信息、收集资料
- 写作者 Agent：整理信息、生成文章

=== 三种协作模式 ===

1. 串行（Sequential）：研究员 → 写作者，按顺序执行
2. 并行（Parallel）：两个 Agent 同时工作，最后合并结果
3. 层级（Hierarchical）：管理者 Agent 分配任务给执行 Agent

=== 本文件演示 ===

串行模式：研究员搜索 → 写作者基于搜索结果生成文章
并行模式：同时搜索 + 同时查阅文件 → 合并结果
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
# Agent 1：研究员 — 收集资料
# ================================================================
RESEARCHER_PROMPT = """你是一个信息研究员。你的职责是：
1. 根据用户的问题，搜索和收集相关信息
2. 整理成结构化的研究笔记
3. 只提供事实，不做主观评价
4. 如果没有找到相关信息，如实说明"""


# ================================================================
# Agent 2：写作者 — 整理生成内容
# ================================================================
WRITER_PROMPT = """你是一个内容写作者。你的职责是：
1. 根据研究员提供的资料，撰写清晰易懂的回答
2. 用结构化的格式呈现（标题、段落、列表）
3. 保持原文的事实准确性，不要添加研究笔记中不存在的信息
4. 使用中文输出"""


# ================================================================
# Agent 3：评审者 — 审核内容质量
# ================================================================
REVIEWER_PROMPT = """你是一个内容评审者。你的职责是：
1. 检查写作者的内容是否基于研究笔记的事实
2. 发现事实错误或遗漏时指出
3. 评估内容质量并给出评分（1-10）
4. 如果内容合格，回复"通过"；否则回复"需要修改"并说明原因"""


# ================================================================
# 串行模式：研究员 → 写作者 → 评审者
# ================================================================
def sequential_multi_agent(topic: str):
    """三个 Agent 串行协作。

    流程：研究员收集资料 → 写作者生成文章 → 评审者审核
    每个阶段依赖前一个阶段的输出。
    """
    print("=" * 60)
    print(f"串行多 Agent 协作")
    print(f"主题: {topic}")
    print("=" * 60)

    # --- 阶段 1：研究员 ---
    print("\n[研究员] 正在收集资料...")
    research_msg = [
        {"role": "system", "content": RESEARCHER_PROMPT},
        {"role": "user", "content": f"请收集关于以下主题的信息，整理成结构化的研究笔记：{topic}"},
    ]
    research_resp = client.chat.completions.create(model=MODEL, messages=research_msg)
    research_notes = research_resp.choices[0].message.content
    print(f"[研究员] 研究笔记（前 300 字）:\n{research_notes[:300]}...\n")

    # --- 阶段 2：写作者 ---
    print("[写作者] 正在撰写文章...")
    writer_msg = [
        {"role": "system", "content": WRITER_PROMPT},
        {"role": "user", "content": f"请根据以下研究笔记，撰写一篇结构清晰的文章：\n\n研究笔记:\n{research_notes}"},
    ]
    writer_resp = client.chat.completions.create(model=MODEL, messages=writer_msg)
    article = writer_resp.choices[0].message.content
    print(f"[写作者] 文章（前 300 字）:\n{article[:300]}...\n")

    # --- 阶段 3：评审者 ---
    print("[评审者] 正在审核...")
    reviewer_msg = [
        {"role": "system", "content": REVIEWER_PROMPT},
        {"role": "user", "content": f"请审核以下内容：\n\n研究笔记:\n{research_notes}\n\n文章:\n{article}"},
    ]
    reviewer_resp = client.chat.completions.create(model=MODEL, messages=reviewer_msg)
    review = reviewer_resp.choices[0].message.content
    print(f"[评审者] 评审结果:\n{review}")

    return {"research": research_notes, "article": article, "review": review}


# ================================================================
# 并行模式：两个 Agent 同时工作
# ================================================================
def parallel_multi_agent(topic: str):
    """两个 Agent 并行工作，然后合并结果。

    流程：
    researcher_A → 搜索技术方面
    researcher_B → 搜索应用方面
           ↓
    writer → 合并两个研究结果，生成综合文章
    """
    print("\n" + "=" * 60)
    print(f"并行多 Agent 协作")
    print(f"主题: {topic}")
    print("=" * 60)

    # --- 并行阶段：两个研究员同时工作 ---
    print("\n[研究员 A] 搜索技术原理方面...")
    notes_a = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "你是技术原理研究员，专门研究'是什么'和'怎么实现'。"},
            {"role": "user", "content": f"请从技术原理角度研究：{topic}"},
        ],
    ).choices[0].message.content

    print("[研究员 B] 搜索应用场景方面...")
    notes_b = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "你是应用场景研究员，专门研究'什么时候用'和'怎么落地'。"},
            {"role": "user", "content": f"请从应用场景角度研究：{topic}"},
        ],
    ).choices[0].message.content

    print(f"  A 产出: {notes_a[:120]}...")
    print(f"  B 产出: {notes_b[:120]}...")

    # --- 合并阶段：写作者整合 ---
    print("\n[写作者] 正在合并两份研究结果...")
    final_article = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "你是一个综合写作者。将两份研究笔记合并成一篇完整的文章。包含'技术原理'和'应用场景'两部分。"},
            {"role": "user", "content": f"研究笔记 A（技术原理）:\n{notes_a}\n\n研究笔记 B（应用场景）:\n{notes_b}"},
        ],
    ).choices[0].message.content

    print(f"[写作者] 综合文章（前 400 字）:\n{final_article[:400]}...")
    return {"tech_research": notes_a, "app_research": notes_b, "final": final_article}


# ================================================================
# 演示
# ================================================================
if __name__ == "__main__":
    # 串行模式
    sequential_multi_agent("AI Agent 在企业中的应用现状")

    # 并行模式
    parallel_multi_agent("RAG 检索增强生成")
