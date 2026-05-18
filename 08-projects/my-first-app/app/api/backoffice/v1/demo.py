"""Backoffice Demo API — 示例后台接口。"""
from fastapi import APIRouter
from app.schemas.response import ApiResponse

router = APIRouter()


@router.get("/health")
async def health_check():
    """健康检查。"""
    return ApiResponse.success(data={"status": "ok"})
