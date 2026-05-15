"""RAG Demo — index documents and ask questions about them."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rag_engine import VectorStore, ingest_directory, rag_query
from dotenv import load_dotenv

load_dotenv()


def demo_rag():
    store = VectorStore()

    # Step 1: Ingest documents
    docs_dir = r"D:\实验"
    print("=" * 60)
    print("Step 1: Indexing documents...")
    print("=" * 60)
    ingest_directory(store, docs_dir, patterns=["*.md"])

    print(f"\nVector store contains {store.count()} chunks.\n")

    # Step 2: Ask questions
    questions = [
        "AI应用开发需要学哪些核心内容？",
        "RAG是什么，大概要学多久？",
        "求职的时候应该准备什么项目？",
        "Python基础学习路线里讲了什么？",
    ]

    for q in questions:
        print("=" * 60)
        print(f"Q: {q}")
        print("=" * 60)

        result = rag_query(store, q, top_k=3)

        print(f"\n{result['answer']}\n")
        print("Sources:")
        for s in result["sources"]:
            print(f"  [{s['source']}] score={s['score']:.3f} — {s['content'][:80]}...")
        print()


def interactive():
    """Interactive Q&A mode."""
    store = VectorStore()
    docs_dir = r"D:\实验"
    print("Indexing documents...")
    ingest_directory(store, docs_dir, patterns=["*.md", "*.txt"])

    print(f"\nLoaded {store.count()} chunks. Ask anything! (type /quit to exit)\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ("/quit", "/q", "/exit"):
            break
        if not q:
            continue

        result = rag_query(store, q, top_k=3)
        print(f"\n{result['answer']}\n")


if __name__ == "__main__":
    demo_rag()
    # Uncomment for interactive mode:
    # interactive()
