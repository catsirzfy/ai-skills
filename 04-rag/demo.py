"""RAG 演示 — 索引本地文档并问答。

运行本文件前请确保 .env 中配置了 DEEPSEEK_API_KEY。
首次运行会调用 Embedding API 索引文档（消耗少量 Token），
之后向量数据保存在 vector_store.json 中，下次运行无需重新索引。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rag_engine import VectorStore, ingest_directory, rag_query
from dotenv import load_dotenv

load_dotenv()


def demo_rag():
    """完整的 RAG 演示流程。"""
    store = VectorStore()

    # ---- 第 1 步：索引文档 ----
    # 把 D:\实验 目录下的所有 .md 文件读入向量库
    docs_dir = r"D:\实验"
    print("=" * 60)
    print("第 1 步：索引文档...")
    print("=" * 60)
    ingest_directory(store, docs_dir, patterns=["*.md"])

    print(f"\n向量库当前包含 {store.count()} 个文本块。\n")

    # ---- 第 2 步：提问 ----
    # AI 会基于刚才索引的文档内容来回答
    questions = [
        "AI应用开发需要学哪些核心内容？",
        "RAG是什么，大概要学多久？",
        "求职的时候应该准备什么项目？",
        "Python基础学习路线里讲了什么？",
    ]

    for q in questions:
        print("=" * 60)
        print(f"问题: {q}")
        print("=" * 60)

        result = rag_query(store, q, top_k=3)  # 每次取最相关的 3 个文档块

        print(f"\n{result['answer']}\n")
        print("参考来源：")
        for s in result["sources"]:
            # 显示相似度分数和文档块前 80 字
            print(f"  [{s['source']}] 相似度={s['score']:.3f} | {s['content'][:80]}...")
        print()


def interactive():
    """交互式 RAG 问答（取消注释启动）。"""
    store = VectorStore()
    docs_dir = r"D:\实验"
    print("索引文档...")
    ingest_directory(store, docs_dir, patterns=["*.md", "*.txt"])
    print(f"\n已加载 {store.count()} 个文本块。开始提问！（输入 /quit 退出）\n")

    while True:
        q = input("你: ").strip()
        if q.lower() in ("/quit", "/q", "/exit"):
            break
        if not q:
            continue
        result = rag_query(store, q, top_k=3)
        print(f"\n{result['answer']}\n")


if __name__ == "__main__":
    demo_rag()
    # 要体验交互式问答，取消下面这行的注释：
    # interactive()
