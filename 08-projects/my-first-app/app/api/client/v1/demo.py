"""Demo API — 示例：替换为你项目的实际 API。"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.api.client.deps import get_current_user
from app.models.user import User
from app.schemas.response import ApiResponse

router = APIRouter()


@router.get("/hello")
async def hello():
    """公开接口 — 无需登录。"""
    return ApiResponse.success(data={"message": "Hello, World!"})


@router.get("/me")
async def me(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """需要登录 — 返回当前用户信息。"""
    return ApiResponse.success(data={
        "id": current_user.id,
        "email": current_user.email,
        "name": current_user.name,
    })
