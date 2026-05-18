"""数据库会话依赖注入。"""
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.base import get_session_local


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖注入：每个请求一个数据库会话。"""
    session_factory = get_session_local()
    async with session_factory() as session:
        yield session
