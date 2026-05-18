"""数据库引擎 + 会话工厂。

用法：
    engine = get_engine()
    AsyncSessionLocal = get_session_local()
    async with AsyncSessionLocal() as session:
        ...
"""
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings

# SQLAlchemy 2.0 声明式基类
class Base(DeclarativeBase):
    pass


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.DATABASE_URL,
            echo=settings.ENV == "development",
            pool_pre_ping=True,
            pool_recycle=1800,
            pool_size=20,
            max_overflow=10,
        )
    return _engine


def get_session_local():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = async_sessionmaker(
            bind=get_engine(), class_=AsyncSession, expire_on_commit=False
        )
    return _SessionLocal


async def close_db():
    if _engine is not None:
        await _engine.dispose()
