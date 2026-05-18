"""AI 对话记录模型。

这是你的第一个自定义 Model。继承 BaseModel 自动获得：
- id        (主键，自增)
- created_at (创建时间)
- updated_at (更新时间)
"""
from sqlalchemy import Column, String, Text, Integer
from app.models.base import BaseModel


class ChatHistory(BaseModel):
    """AI 对话记录表。

    每一行 = 一轮对话（一个问题 + 一个回答）。
    """
    __tablename__ = "chat_history"

    # 用户的提问
    question = Column(Text, nullable=False)

    # AI 的回答
    answer = Column(Text, nullable=False)

    # 使用的模型（如 deepseek-chat）
    model = Column(String(100), default="deepseek-chat")

    # Token 消耗（输入 Token 数）
    tokens_used = Column(Integer, default=0)
