"""
RAG 引擎 — 检索增强生成的完整实现。

=== RAG 是什么 ===

RAG = Retrieval-Augmented Generation（检索增强生成）

传统 LLM 问题：训练数据有截止日期，不知道你的私有文档内容。
RAG 解决方案：先把你的文档"喂"给 AI，让 AI 基于文档内容回答。

=== 完整流程（5 步） ===

  文档(.md/.txt) → 分块(chunking) → 向量化(embedding) → 存入向量库 → 用户提问

  用户提问 → 向量化 → 语义搜索相关块 → 把块 + 问题发给 AI → AI 基于文档回答

=== 为什么需要分块（Chunking） ===

1. AI 的上下文窗口有长度限制（虽然现在越来越长）
2. 块太大：检索精度低，不相关的信息混进来
3. 块太小：语义不完整，AI 看不明白
4. 经验值：500 字一块，80 字重叠（保持上下文连贯）

=== 为什么需要 Embedding ===

文本 AI 看不懂，需要转成数字（向量）。
"你好" → [0.2, 0.8, -0.1, 0.5, ...] （通常 1536 维）

语义越相近的文本，向量在空间中越靠近。
用"余弦相似度"衡量两个向量的接近程度。

=== 为什么用 numpy 而不是 Chroma/Milvus ===

学习阶段用 numpy 可以看到向量搜索的内部细节。
生产环境当然用专业的向量数据库——但原理完全一样。
"""

import os
import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 必须在创建 HuggingFace 模型之前设置（解决国内网络问题）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Python 3.14 的 SSL 证书链可能不完整，禁用验证（仅用于下载模型）
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 分块参数
CHUNK_SIZE = 500      # 每块最大字符数
CHUNK_OVERLAP = 80    # 相邻块之间的重叠字符数（保持上下文连续）

# Embedding 模式：local=本地模型(免费,离线) | api=调用Embedding API
EMBED_MODE = "local"

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "your-deepseek-key"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

# 本地 Embedding 模型（懒加载，首次运行会自动下载 ~90MB）
_embed_model = None


def _get_embed_model():
    """懒加载本地 embedding 模型。

    使用 BAAI/bge-small-zh — 中文专用，轻量（90MB），无需 GPU。
    首次运行自动下载（通过国内镜像 hf-mirror.com）。
    """
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    return _embed_model


# ================================================================
# 第 1 步：文档加载
# ================================================================
def load_file(path: str) -> str:
    """从文件读取纯文本内容。

    生产环境需要支持 PDF、Word 等格式，这里先支持最常见的文本格式。
    扩展时可以用 langchain_community.document_loaders 或 PyPDF2。
    """
    ext = Path(path).suffix.lower()
    if ext not in (".md", ".txt", ".py", ".json", ".yaml", ".yml", ".csv"):
        raise ValueError(f"不支持的格式: {ext}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ================================================================
# 第 2 步：文本分块
# ================================================================
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """将长文本切分成带重叠的小块。

    策略：先按段落边界切分，超长的段落再按字符数切分。
    这样每个 chunk 通常在语义上是完整的段落。
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            # 当前块还有空间，追加段落
            current += ("\n\n" if current else "") + para
        else:
            # 当前块满了，保存并开始新块
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                # 单个段落太长，按字符切分（带重叠）
                for i in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[i:i + chunk_size])
            else:
                current = para

    if current:
        chunks.append(current)
    return chunks


# ================================================================
# 第 3 步：文本向量化（Embedding）
# ================================================================
def embed_texts(texts: list[str]) -> np.ndarray:
    """将文本列表转换为向量矩阵。

    两种模式：
    - local: 使用 BAAI/bge-small-zh 本地模型（免费、离线、90MB）
    - api:   使用 OpenAI 兼容的 Embedding API
    """
    if EMBED_MODE == "local":
        model = _get_embed_model()
        # sentence-transformers 的 encode 方法，返回 numpy 数组
        return model.encode(texts, normalize_embeddings=True)

    # API 模式
    all_embeddings = []
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        all_embeddings.extend([d.embedding for d in response.data])
    return np.array(all_embeddings, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """将单个查询文本向量化。"""
    return embed_texts([text])[0]


# ================================================================
# 第 4 步：向量存储 & 语义搜索
# ================================================================
class VectorStore:
    """基于 numpy 的轻量级向量存储。

    用余弦相似度做语义搜索：
    - 把查询变成向量
    - 和库里所有文档向量算相似度
    - 取最相似的 top_k 个

    持久化到 JSON 文件，重启后数据不丢失。

    生产环境中这里会用 Chroma/Milvus/Pinecone 等专业向量数据库，
    但核心逻辑（embedding + 余弦相似度）完全一样。
    """

    def __init__(self, save_path: str = None):
        self.save_path = save_path or os.path.join(os.path.dirname(__file__), "vector_store.json")
        self.chunks: list[str] = []      # 文档块原文
        self.sources: list[str] = []     # 每个块的来源文件名
        self.vectors: np.ndarray | None = None  # 向量矩阵，每行一个 chunk

        if os.path.exists(self.save_path):
            self._load()

    def add(self, chunks: list[str], source: str = "unknown"):
        """将文档块向量化并存入。"""
        if not chunks:
            return 0
        vectors = embed_texts(chunks)

        self.chunks.extend(chunks)
        self.sources.extend([source] * len(chunks))

        if self.vectors is None:
            self.vectors = vectors
        else:
            # vstack：垂直拼接两个矩阵
            self.vectors = np.vstack([self.vectors, vectors])

        self._save()
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """余弦相似度语义搜索。

        公式：cos_sim(a, b) = dot(a, b) / (|a| * |b|)
        结果越接近 1 越相似。
        """
        if self.vectors is None or len(self.chunks) == 0:
            return []

        qvec = embed_query(query)

        # 计算查询向量和所有文档向量之间的余弦相似度
        # norms: 每个文档向量的 L2 范数（长度）
        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(qvec)
        # similarities: 余弦相似度数组，长度 = chunk 数量
        similarities = np.dot(self.vectors, qvec) / (norms + 1e-10)

        # argsort 返回从小到大的索引，[::-1] 反转，取前 top_k
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
        """保存到 JSON 文件（向量转成 list 才能序列化）。"""
        data = {
            "chunks": self.chunks,
            "sources": self.sources,
            "vectors": self.vectors.tolist() if self.vectors is not None else [],
        }
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def _load(self):
        """从 JSON 文件恢复。"""
        with open(self.save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = data["chunks"]
        self.sources = data["sources"]
        self.vectors = np.array(data["vectors"], dtype=np.float32) if data["vectors"] else None


# ================================================================
# 第 5 步：RAG 查询 — 把前面四步串起来
# ================================================================
def rag_query(store: VectorStore, question: str, top_k: int = 5) -> dict:
    """完整的 RAG 查询：检索 → 增强 → 生成。

    1. 检索（Retrieve）：从向量库找到最相关的文档块
    2. 增强（Augment）：把文档块拼接成上下文
    3. 生成（Generate）：把上下文 + 问题发给 AI，得到基于文档的回答
    """
    hits = store.search(question, top_k=top_k)

    if not hits:
        return {"answer": "知识库中没有找到相关信息。", "sources": []}

    # 拼接上下文 — 把检索到的文档块用分隔符连起来
    context = "\n\n---\n\n".join(
        f"[来源: {h['source']}] {h['content']}" for h in hits
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个知识库问答助手。请根据提供的文档内容回答问题。"
                    "如果文档中没有相关信息，请如实告知。回答时引用来源。"
                ),
            },
            # 关键：把检索到的文档作为上下文注入到 user 消息中
            {"role": "user", "content": f"文档内容：\n{context}\n\n问题：{question}"},
        ],
        temperature=0.3,  # 低温让回答更准确，减少编造（幻觉）
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": hits,  # 返回引用来源，方便验证
    }


def ingest_directory(store: VectorStore, dir_path: str, patterns: list[str] = None):
    """批量索引目录下所有匹配的文件。"""
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
            print(f"  已索引: {fp.name} ({n} 块)")
            total += n
        except Exception as e:
            print(f"  跳过: {fp.name} — {e}")

    print(f"\n总计: {total} 个文本块，来自 {len(files)} 个文件")
    return total
