"""
RAG API Server — FastAPI + SSE 流式输出。

=== 为什么要 FastAPI ===

1. 原生支持异步（async/await），AI 调用本身就是 IO 密集型
2. 内置 SSE（Server-Sent Events）支持，流式输出开箱即用
3. 自动生成 API 文档（/docs），面试时可以直接展示
4. Python 后端岗位标配框架

=== SSE（Server-Sent Events）原理 ===

普通 HTTP：客户端请求 → 服务器一次返回 → 连接关闭
SSE：      客户端请求 → 服务器保持连接 → 持续推送数据流

对于 AI 应用：
- 普通模式：等 5 秒 → 一次性返回完整答案（用户盯着白屏）
- SSE 模式：每秒推送几个字 → 用户看到打字效果（体验好）

=== 启动方式 ===

    uvicorn api_server:app --reload

然后打开浏览器访问 http://localhost:8000/docs 看 API 文档
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from rag_engine import (
    VectorStore, ingest_directory,
    hybrid_search, rerank_results,
    rag_query, MultiTurnRAG,
)

# --- 初始化 ---
app = FastAPI(
    title="RAG Knowledge Base API",
    description="检索增强生成知识库问答系统",
    version="1.0.0",
)

# 启动时自动索引文档
store = VectorStore()
multiturn = MultiTurnRAG(store)

print("正在索引文档...")
ingest_directory(store, r"D:\实验", patterns=["*.md"])
print(f"索引完成：{store.count()} 个文本块")


# --- 数据模型 ---
class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5
    use_hybrid: bool = True     # 是否使用混合检索
    use_rerank: bool = False     # 是否使用重排序
    stream: bool = True          # 是否流式输出
    session_id: Optional[str] = None  # 会话 ID（多轮对话）


class IngestRequest(BaseModel):
    """上传文档的请求体。"""
    text: str
    source: str = "api_upload"


# --- API 端点 ---
@app.get("/")
def root():
    return {
        "service": "RAG Knowledge Base API",
        "docs": "/docs",
        "endpoints": ["POST /chat", "POST /chat/stream", "POST /ingest", "GET /stats"],
    }


@app.post("/chat")
def chat(req: QuestionRequest):
    """普通问答（一次性返回）。适合后端调用、批量处理。"""
    if req.use_hybrid:
        hits = hybrid_search(store, req.question, top_k=req.top_k)
    else:
        hits = store.search(req.question, top_k=req.top_k)

    if req.use_rerank and len(hits) > 3:
        hits = rerank_results(req.question, hits, top_k=req.top_k)

    if not hits:
        return {"answer": "知识库中没有找到相关信息。", "sources": []}

    # 拼接上下文并调用 LLM
    context = "\n\n---\n\n".join(
        f"[来源: {h['source']}] {h['content']}" for h in hits
    )

    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()

    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个知识库问答助手。根据文档内容回答问题。如果文档中没有相关信息，如实告知。"},
            {"role": "user", "content": f"文档内容：\n{context}\n\n问题：{req.question}"},
        ],
        temperature=0.3,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": [{"content": h["content"][:100], "source": h["source"], "score": h["score"]} for h in hits],
    }


@app.post("/chat/stream")
def chat_stream(req: QuestionRequest):
    """流式问答（SSE）。前端实时看到 AI 逐字生成。

    这是面试最加分的端点——展示你理解了 SSE 协议。
    """
    if req.use_hybrid:
        hits = hybrid_search(store, req.question, top_k=req.top_k)
    else:
        hits = store.search(req.question, top_k=req.top_k)

    if req.use_rerank and len(hits) > 3:
        hits = rerank_results(req.question, hits, top_k=req.top_k)

    if not hits:
        def no_result():
            yield "data: " + json.dumps({"error": "知识库中没有找到相关信息。"}, ensure_ascii=False) + "\n\n"
        return StreamingResponse(no_result(), media_type="text/event-stream")

    context = "\n\n---\n\n".join(
        f"[来源: {h['source']}] {h['content']}" for h in hits
    )

    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()

    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )

    def generate():
        """SSE 生成器 — 每个 chunk 是一条 event。"""
        stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个知识库问答助手。根据文档内容回答问题。"},
                {"role": "user", "content": f"文档内容：\n{context}\n\n问题：{req.question}"},
            ],
            temperature=0.3,
            stream=True,
        )

        # SSE 格式：data: {json}\n\n
        for chunk in stream:
            if content := chunk.choices[0].delta.content:
                yield f"data: {json.dumps({'text': content}, ensure_ascii=False)}\n\n"

        # 发送完成信号
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        },
    )


@app.post("/ingest")
def ingest(req: IngestRequest):
    """上传新文档到知识库。"""
    from rag_engine import chunk_text

    chunks = chunk_text(req.text)
    n = store.add(chunks, source=req.source)
    return {"status": "ok", "chunks_added": n, "total_chunks": store.count()}


@app.get("/stats")
def stats():
    """知识库统计信息。"""
    return {
        "total_chunks": store.count(),
        "indexed_files": list(store.indexed_files) if hasattr(store, 'indexed_files') else [],
    }


# --- 启动命令 ---
# uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
#
# 然后在浏览器打开:
#   http://localhost:8000/docs     — 交互式 API 文档
#   http://localhost:8000/         — 服务信息
#
# 测试流式输出:
#   curl -X POST http://localhost:8000/chat/stream \
#     -H "Content-Type: application/json" \
#     -d '{"question": "RAG要学多久？", "use_hybrid": true, "stream": true}'
