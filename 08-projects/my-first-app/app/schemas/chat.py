"""对话相关的 Pydantic Schema。

Schema 的作用：
- 定义 API 接收什么参数（请求体校验）
- 定义 API 返回什么格式（响应体格式化）
- FastAPI 自动根据 Schema 生成 API 文档
"""
from pydantic import BaseModel
from typing import Optional


class ChatCreate(BaseModel):
    """创建对话记录的请求体。"""
    question: str
    answer: str
    model: str = "deepseek-chat"
    tokens_used: int = 0


class ChatResponse(BaseModel):
    """对话记录的响应体。"""
    id: int
    question: str
    answer: str
    model: str
    tokens_used: int

    class Config:
        from_attributes = True  # 允许从 ORM 对象转换
