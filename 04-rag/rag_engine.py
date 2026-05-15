"""RAG Engine — lightweight implementation, zero heavy dependencies.

Pipeline:
    Documents → Chunking → Embedding → Vector Store → Search → AI Answer

Uses numpy for vector math, DeepSeek API for embeddings and chat.
"""

import os
import sys
import json
import math
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "your-deepseek-key"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)


# ================================================================
# Document loading
# ================================================================
def load_file(path: str) -> str:
    """Load text from .md, .txt, .py, .json files."""
    ext = Path(path).suffix.lower()
    if ext not in (".md", ".txt", ".py", ".json", ".yaml", ".yml", ".csv"):
        raise ValueError(f"Unsupported format: {ext}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ================================================================
# Chunking
# ================================================================
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks at paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current += ("\n\n" if current else "") + para
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[i:i + chunk_size])
            else:
                current = para

    if current:
        chunks.append(current)
    return chunks


# ================================================================
# Embedding
# ================================================================
def embed_texts(texts: list[str]) -> np.ndarray:
    """Convert texts to vectors. Batches to avoid API limits."""
    all_embeddings = []
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model="text-embedding-3-small", input=batch)
        all_embeddings.extend([d.embedding for d in response.data])
    return np.array(all_embeddings, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]


# ================================================================
# Vector Store (numpy-based, saved as JSON)
# ================================================================
class VectorStore:
    """Simple vector store with cosine similarity search. Persists to disk."""

    def __init__(self, save_path: str = None):
        self.save_path = save_path or os.path.join(os.path.dirname(__file__), "vector_store.json")
        self.chunks: list[str] = []
        self.sources: list[str] = []
        self.vectors: np.ndarray | None = None

        if os.path.exists(self.save_path):
            self._load()

    def add(self, chunks: list[str], source: str = "unknown"):
        """Embed and store chunks."""
        if not chunks:
            return 0
        vectors = embed_texts(chunks)

        self.chunks.extend(chunks)
        self.sources.extend([source] * len(chunks))

        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

        self._save()
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Cosine similarity search."""
        if self.vectors is None or len(self.chunks) == 0:
            return []

        qvec = embed_query(query)
        # Cosine similarity
        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(qvec)
        similarities = np.dot(self.vectors, qvec) / (norms + 1e-10)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            {
                "content": self.chunks[i],
                "source": self.sources[i],
                "score": float(similarities[i]),
            }
            for i in top_indices
        ]

    def count(self) -> int:
        return len(self.chunks)

    def clear(self):
        self.chunks = []
        self.sources = []
        self.vectors = None
        if os.path.exists(self.save_path):
            os.remove(self.save_path)

    def _save(self):
        data = {
            "chunks": self.chunks,
            "sources": self.sources,
            "vectors": self.vectors.tolist() if self.vectors is not None else [],
        }
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def _load(self):
        with open(self.save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = data["chunks"]
        self.sources = data["sources"]
        self.vectors = np.array(data["vectors"], dtype=np.float32) if data["vectors"] else None


# ================================================================
# RAG Query
# ================================================================
def rag_query(store: VectorStore, question: str, top_k: int = 5) -> dict:
    """Complete RAG pipeline: retrieve → augment → generate."""
    hits = store.search(question, top_k=top_k)

    if not hits:
        return {"answer": "知识库中没有找到相关信息。", "sources": []}

    context = "\n\n---\n\n".join(
        f"[来源: {h['source']}] {h['content']}" for h in hits
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个知识库问答助手。根据提供的文档内容回答问题。"
                    "如果文档中没有相关信息，如实告知。回答时引用来源。"
                ),
            },
            {"role": "user", "content": f"文档内容：\n{context}\n\n问题：{question}"},
        ],
        temperature=0.3,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": hits,
    }


def ingest_directory(store: VectorStore, dir_path: str, patterns: list[str] = None):
    """Index all matching files in a directory."""
    if patterns is None:
        patterns = ["*.md", "*.txt"]

    files = []
    for p in patterns:
        files.extend(Path(dir_path).glob(p))

    total = 0
    for fp in files:
        try:
            text = load_file(str(fp))
            chunks = chunk_text(text)
            n = store.add(chunks, source=fp.name)
            print(f"  Indexed: {fp.name} ({n} chunks)")
            total += n
        except Exception as e:
            print(f"  Skip: {fp.name} — {e}")

    print(f"\nTotal: {total} chunks from {len(files)} files")
    return total
