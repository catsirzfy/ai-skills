"""模型基类 — 所有数据库表继承这个。"""
from sqlalchemy import Column, Integer, TIMESTAMP, func
from app.db.base import Base


class BaseModel(Base):
    """自带 id + created_at + updated_at 的模型基类。"""
    __abstract__ = True

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
