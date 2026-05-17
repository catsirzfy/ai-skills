"""RAG 高级特性演示：混合检索、重排序、多轮对话。"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from rag_engine import (
    VectorStore, ingest_directory,
    hybrid_search, rerank_results, MultiTurnRAG,
    rag_query,
)
from dotenv import load_dotenv
load_dotenv()


def demo_hybrid_search():
    """演示混合检索 vs 纯语义搜索。

    找一个包含特定关键词的问题，对比两种检索结果。
    """
    print("=" * 60)
    print("高级特性 1：混合检索（语义 + 关键词）")
    print("=" * 60)

    store = VectorStore()
    ingest_directory(store, r"D:\实验", patterns=["*.md"])

    query = "Agent 要学多久？"

    print("\n--- 纯语义搜索 ---")
    hits = store.search(query, top_k=3)
    for h in hits:
        print(f"  [{h['source']}] score={h['score']:.3f} | {h['content'][:80]}...")

    print("\n--- 混合检索 ---")
    hits = hybrid_search(store, query, top_k=3, alpha=0.7)
    for h in hits:
        print(f"  [{h['source']}] combined={h['score']:.3f} (语义={h['semantic_score']:.3f}, 关键词={h['keyword_score']:.3f}) | {h['content'][:80]}...")


def demo_reranking():
    """演示重排序的效果。

    初检取 10 条，用 Cross-encoder 重新排序，看前后变化。
    """
    print("\n" + "=" * 60)
    print("高级特性 2：重排序（Cross-encoder Reranker）")
    print("=" * 60)

    store = VectorStore()
    query = "Python 后端工程师需要学什么？"

    # 初检取 10 条
    candidates = store.search(query, top_k=10)
    print(f"\n初检 top 3:")
    for h in candidates[:3]:
        print(f"  [{h['source']}] score={h['score']:.3f} | {h['content'][:80]}...")

    # 重排序取 3 条
    print("\n正在重排序...")
    ranked = rerank_results(query, candidates, top_k=3)

    print(f"\n重排序后 top 3:")
    for h in ranked:
        print(f"  [{h['source']}] rerank={h['rerank_score']:.3f} (原={h['score']:.3f}) | {h['content'][:80]}...")
    print("\n注意：重排序后的顺序和初检可能不同——Cross-encoder 更精准。")


def demo_multiturn_rag():
    """演示多轮对话 RAG。

    先问一个具体问题，再问一个模糊的追问，看 AI 能否理解上下文。
    """
    print("\n" + "=" * 60)
    print("高级特性 3：多轮对话 RAG")
    print("=" * 60)

    store = VectorStore()

    rag = MultiTurnRAG(store)

    # 第 1 轮：正常问题
    q1 = "AI应用开发的RAG需要学哪些内容？"
    print(f"\n第 1 轮: {q1}")
    result = rag.ask(q1, top_k=3)
    # 提取纯文本回答（去掉引用标记）
    answer = result["answer"].split("[来源:")[0].strip()
    print(f"改写: {result['rewritten_query']}")
    print(f"回答: {answer[:200]}...")

    # 第 2 轮：模糊追问
    q2 = "那大概要花多少时间？"
    print(f"\n第 2 轮: {q2}")
    result = rag.ask(q2, top_k=3)
    answer = result["answer"].split("[来源:")[0].strip()
    print(f"改写: {result['rewritten_query']}")
    print(f"回答: {answer[:200]}...")

    # 第 3 轮：更模糊的追问
    q3 = "除了这个还有什么核心的？"
    print(f"\n第 3 轮: {q3}")
    result = rag.ask(q3, top_k=3)
    answer = result["answer"].split("[来源:")[0].strip()
    print(f"改写: {result['rewritten_query']}")
    print(f"回答: {answer[:200]}...")

    print("\n关键观察：每轮的 rewritten_query 把模糊追问补全成了完整问题。")


if __name__ == "__main__":
    demo_hybrid_search()
    demo_reranking()
    demo_multiturn_rag()
