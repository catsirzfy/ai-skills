"""客户端 API 依赖注入。

用法：
    current_user: User = Depends(get_current_user)
"""
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.config import settings
from app.core.security import verify_token
from app.db.session import get_db
from app.models.user import User
from app.exceptions import AuthenticationError

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """验证 JWT 并返回当前用户。"""
    payload = verify_token(token, scope="client")
    if not payload:
        raise AuthenticationError(message="无效的认证凭据")

    user_id = payload.get("sub")
    result = await db.execute(select(User).where(User.id == int(user_id)))
    user = result.scalar_one_or_none()

    if not user:
        raise AuthenticationError(message="用户不存在")
    if not user.is_active:
        raise AuthenticationError(message="用户已被禁用")
    return user
