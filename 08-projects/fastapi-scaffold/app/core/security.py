"""认证模块 — JWT 生成 + 验证 + 密码哈希。"""
from datetime import datetime, timedelta, UTC
from typing import Optional
import uuid

from jose import jwt, JWTError
from passlib.context import CryptContext

from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """密码哈希。"""
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """验证密码。"""
    return pwd_context.verify(plain, hashed)


def create_access_token(user_id: int, scope: str = "client") -> str:
    """生成 Access Token（短期，通常 30 分钟）。"""
    expire = datetime.now(UTC) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "exp": expire,
        "sub": str(user_id),
        "scope": scope,
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(user_id: int) -> str:
    """生成 Refresh Token（长期，通常 7 天）。"""
    expire = datetime.now(UTC) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "exp": expire,
        "sub": str(user_id),
        "scope": "refresh",
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def verify_token(token: str, scope: str = None) -> Optional[dict]:
    """验证 Token 并返回 payload，失败返回 None。"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        if scope and payload.get("scope") != scope:
            return None
        return payload
    except JWTError:
        return None
