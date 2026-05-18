"""用户模型 — 示例：替换为你项目的实际模型。"""
from sqlalchemy import Column, String, Boolean
from app.models.base import BaseModel


class User(BaseModel):
    __tablename__ = "users"

    email = Column(String(255), unique=True, index=True, nullable=False)
    password = Column(String(255), nullable=False)
    name = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
