"""
LlamaIndex RAG Demo — 用 LlamaIndex 框架实现知识库问答。

=== LlamaIndex 的设计哲学 ===

"Data framework for LLM applications" — 数据是核心。
LlamaIndex 把一切都抽象成 Document → Node → Index 的数据流。

核心理念：
  你的数据(Document) → 解析成节点(Node) → 建立索引(Index) → 问答

=== LlamaIndex 的核心对象 ===

1. Document — 代表一份文档（文件内容 + 元数据）
2. Node — 文档切分后的最小单元（类似我们的 chunk）
3. VectorStoreIndex — 向量索引，负责 embedding + 存储 + 检索
4. QueryEngine — 查询引擎，把检索 + LLM 调用封装在一起

一句话：你给 LlamaIndex 一堆 Document，它帮你做完剩下的所有事。

=== 和手写 RAG 的对比 ===

手写版做的事               LlamaIndex 对应的封装
─────────────────────────────────────────────────
load_file()               SimpleDirectoryReader
chunk_text()              SentenceSplitter
embed_texts()             OpenAIEmbedding / HuggingFaceEmbedding
VectorStore.search()      VectorStoreIndex.as_query_engine()
rag_query()               query_engine.query()
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# --- 必须在 import HuggingFace 模型之前设置镜像 ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from dotenv import load_dotenv
load_dotenv()

# --- 配置 DeepSeek 作为 LLM ---
# LlamaIndex 的 OpenAI 类会校验模型名是否在 OpenAI 已知列表中
# 需要 monkeypatch 两个地方：utils 模块和 base 模块（各自 import 了该函数）
import llama_index.llms.openai.utils as _utils
import llama_index.llms.openai.base as _base

_orig = _utils.openai_modelname_to_contextsize

def _patched_contextsize(name):
    try:
        return _orig(name)
    except ValueError:
        return 65536  # DeepSeek 上下文窗口大小

_utils.openai_modelname_to_contextsize = _patched_contextsize
_base.openai_modelname_to_contextsize = _patched_contextsize

from llama_index.llms.openai import OpenAI as LlamaOpenAI

llm = LlamaOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY", "your-deepseek-key"),
    api_base=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    max_tokens=4096,
)

# DeepSeek 不支持 /v1/completions，只支持 /v1/chat/completions
# 直接替换底层的 _complete 方法，让它走 chat API
_orig_complete_method = LlamaOpenAI._complete

def _patched_complete(self, prompt, formatted=False, **kwargs):
    """用 chat API 实现 completion（绕过 DeepSeek 不支持 /completions）。"""
    client = self._get_client()
    response = client.chat.completions.create(
        model=self.model,
        messages=[{"role": "user", "content": prompt}],
        **{k: v for k, v in kwargs.items() if k not in ("prompt", "formatted")},
    )
    from llama_index.core.base.llms.types import CompletionResponse
    return CompletionResponse(text=response.choices[0].message.content)

LlamaOpenAI._complete = _patched_complete

# --- 配置本地 Embedding 模型 ---
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5",
)

# 全局配置：告诉 LlamaIndex 用哪个 LLM 和 Embedding
from llama_index.core import Settings
Settings.llm = llm
Settings.embed_model = embed_model


def llamaindex_rag():
    """完整的 LlamaIndex RAG 流程。

    下面 4 步对应手写版的 5 步，你会发现 4 步 vs 200 行代码的区别。
    """
    from llama_index.core import (
        VectorStoreIndex,
        SimpleDirectoryReader,
    )

    # 第 1+2 步：加载文档 + 自动分块
    # SimpleDirectoryReader 自动识别文件类型并读取
    # 它内部会用 SentenceSplitter 自动分块（默认 chunk_size=1024）
    print("=" * 60)
    print("第 1+2 步：加载文档并建立索引...")
    print("=" * 60)

    documents = SimpleDirectoryReader(
        input_dir=r"D:\实验",
        required_exts=[".md"],      # 只读 markdown 文件
        recursive=False,
    ).load_data()

    print(f"  加载了 {len(documents)} 个文档")
    for doc in documents:
        print(f"    - {doc.metadata.get('file_name', '?')}")

    # 第 3+4 步：建立向量索引
    # from_documents() 内部做：分块 → embedding → 存入向量库
    # 这一行代码替代了我们手写的大约 100 行
    print("\n正在建立向量索引（分块 + embedding + 入库）...")
    index = VectorStoreIndex.from_documents(documents)

    # 第 5 步：创建查询引擎并提问
    # as_query_engine() 返回一个可以直接 .query() 的对象
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        response_mode="compact",  # 用 compact 模式，走 chat API 而非 completions API
    )

    questions = [
        "AI应用开发需要学哪些核心内容？",
        "RAG是什么，大概要学多久？",
        "求职的时候应该准备什么项目？",
    ]

    for q in questions:
        print(f"\n{'=' * 60}")
        print(f"问题: {q}")
        print("=" * 60)
        response = query_engine.query(q)
        print(f"\n{response}\n")
        # response.source_nodes 包含检索到的源文档信息
        print("参考来源：")
        for node in response.source_nodes:
            score = node.score if node.score else 0
            source = node.metadata.get("file_name", "unknown")
            print(f"  [{source}] 相似度={score:.3f} | {node.text[:80]}...")
        print()


if __name__ == "__main__":
    if os.getenv("DEEPSEEK_API_KEY", "").startswith("sk-"):
        llamaindex_rag()
    else:
        print("请先配置 .env 中的 DEEPSEEK_API_KEY")
