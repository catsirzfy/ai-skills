"""统一响应格式 — 所有 API 返回结构一致。

返回结构：
    {"code": 200, "message": "Success", "data": {...}}
"""
from typing import Any, Optional
from fastapi import status
from fastapi.responses import JSONResponse, Response
from fastapi.encoders import jsonable_encoder


class ApiResponse:
    """统一 API 响应工具类。"""

    @staticmethod
    def success(data: Any = None, message: str = "Success",
                http_code: int = status.HTTP_200_OK) -> JSONResponse:
        """成功响应。"""
        return JSONResponse(
            content={"code": 200, "message": message,
                     "data": jsonable_encoder(data) if data is not None else None},
            status_code=http_code,
        )

    @staticmethod
    def success_no_content() -> Response:
        """空响应（204）。"""
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @staticmethod
    def error(message: str, code: int = 10000,
              http_code: int = status.HTTP_400_BAD_REQUEST,
              data: Any = None) -> JSONResponse:
        """错误响应。"""
        body = {"code": code, "message": message}
        if data is not None:
            body["data"] = jsonable_encoder(data)
        return JSONResponse(content=body, status_code=http_code)
