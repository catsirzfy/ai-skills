"""对话记录 API + AI 问答。

接口：
    POST /api/v1/chats/ask    — 提问，调 AI，保存记录，返回回答
    GET  /api/v1/chats        — 历史列表（分页）
    GET  /api/v1/chats/{id}   — 查看单条
    DELETE /api/v1/chats/{id} — 删除
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from app.db.session import get_db
from app.schemas.response import ApiResponse
from app.schemas.chat import ChatCreate, ChatResponse
from app.services.chat_service import ChatService
from app.services.ai_service import ask_ai

router = APIRouter()


class AskRequest(BaseModel):
    question: str


@router.post("/ask")
async def ask_question(
    data: AskRequest,
    db: AsyncSession = Depends(get_db),
):
    """用 AI 回答问题并保存记录。

    这是前端"提问"按钮调用的接口。
    流程：接收问题 → 调 DeepSeek → 保存到数据库 → 返回答案。
    """
    # 调 AI 获取回答
    answer, model, tokens = await ask_ai(data.question)

    # 保存到数据库
    record = await ChatService.create(db, ChatCreate(
        question=data.question,
        answer=answer,
        model=model,
        tokens_used=tokens,
    ))

    return ApiResponse.success(data={
        "id": record.id,
        "question": data.question,
        "answer": answer,
        "model": model,
        "tokens_used": tokens,
    })


@router.post("")
async def create_chat(
    data: ChatCreate,
    db: AsyncSession = Depends(get_db),
):
    """手动保存一条对话记录（不用 AI）。"""
    record = await ChatService.create(db, data)
    return ApiResponse.success(data=ChatResponse.from_orm(record).model_dump())


@router.get("")
async def list_chats(
    page: int = Query(1, ge=1),               # ← Query 参数，ge=1 表示 >=1
    per_page: int = Query(20, ge=1, le=100),  # ← le=100 表示 <=100
    db: AsyncSession = Depends(get_db),
):
    """分页查询对话记录。"""
    result = await ChatService.list(db, page, per_page)
    return ApiResponse.success(data={
        "items": [ChatResponse.from_orm(r).model_dump() for r in result["items"]],
        "total": result["total"],
        "page": result["page"],
        "per_page": result["per_page"],
    })


@router.get("/{chat_id}")
async def get_chat(
    chat_id: int,                              # ← 路径参数
    db: AsyncSession = Depends(get_db),
):
    """获取单条对话记录。"""
    record = await ChatService.get(db, chat_id)
    return ApiResponse.success(data=ChatResponse.from_orm(record).model_dump())


@router.delete("/{chat_id}")
async def delete_chat(
    chat_id: int,
    db: AsyncSession = Depends(get_db),
):
    """删除一条对话记录。"""
    await ChatService.delete(db, chat_id)
    return ApiResponse.success(message="删除成功")
