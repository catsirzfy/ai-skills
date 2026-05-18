"""对话记录 Service — 业务逻辑层。

Service 是 Router 和 Database 之间的桥梁：
- Router 负责"接收 HTTP 请求，返回 HTTP 响应"
- Service 负责"业务逻辑，操作数据库"
- Router 不直接写 SQL，而是调用 Service
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.models.chat import ChatHistory
from app.schemas.chat import ChatCreate
from app.exceptions import NotFoundError


class ChatService:

    @staticmethod
    async def create(db: AsyncSession, data: ChatCreate) -> ChatHistory:
        """保存一条 AI 对话记录。"""
        record = ChatHistory(
            question=data.question,
            answer=data.answer,
            model=data.model,
            tokens_used=data.tokens_used,
        )
        db.add(record)
        await db.commit()
        await db.refresh(record)  # 刷新以获取数据库生成的 id 和时间戳
        return record

    @staticmethod
    async def list(db: AsyncSession, page: int = 1, per_page: int = 20):
        """分页获取对话记录列表。"""
        # 计算总数
        total = await db.scalar(select(func.count()).select_from(ChatHistory))

        # 分页查询，按时间倒序（最新的在前）
        offset = (page - 1) * per_page
        query = select(ChatHistory).order_by(ChatHistory.id.desc()).offset(offset).limit(per_page)
        result = await db.execute(query)
        records = result.scalars().all()

        return {
            "items": records,
            "total": total,
            "page": page,
            "per_page": per_page,
        }

    @staticmethod
    async def get(db: AsyncSession, chat_id: int) -> ChatHistory:
        """获取单条对话记录。"""
        record = await db.get(ChatHistory, chat_id)
        if not record:
            raise NotFoundError(message=f"对话记录 {chat_id} 不存在")
        return record

    @staticmethod
    async def delete(db: AsyncSession, chat_id: int):
        """删除一条对话记录。"""
        record = await db.get(ChatHistory, chat_id)
        if not record:
            raise NotFoundError(message=f"对话记录 {chat_id} 不存在")
        await db.delete(record)
        await db.commit()
