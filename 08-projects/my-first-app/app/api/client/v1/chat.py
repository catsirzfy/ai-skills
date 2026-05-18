"""对话记录 API。

一个完整的 CRUD（增删查）示例。
每个接口的模式都一样：
    1. 接收参数（Pydantic 自动校验）
    2. 调 Service（业务逻辑）
    3. 返回 ApiResponse（统一格式）
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.schemas.response import ApiResponse
from app.schemas.chat import ChatCreate, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter()


@router.post("")
async def create_chat(
    data: ChatCreate,                          # ← 请求体自动校验
    db: AsyncSession = Depends(get_db),        # ← 自动注入数据库会话
):
    """保存一条 AI 对话记录。"""
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
