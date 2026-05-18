"""统一异常体系 — 分层错误码，前端根据 code 处理不同错误。"""
from typing import Any, Optional
from fastapi import HTTPException


class APIException(HTTPException):
    """基础异常，所有业务异常继承这个。"""
    def __init__(self, code: int = 10000, message: str = "API 异常",
                 status_code: int = 400, data: Any = None):
        super().__init__(status_code=status_code, detail=message)
        self.code = code
        self.data = data


class ValidationError(APIException):
    def __init__(self, message: str = "参数验证错误", data: Any = None):
        super().__init__(code=1001, message=message, status_code=400, data=data)


class AuthenticationError(APIException):
    def __init__(self, message: str = "认证失败", data: Any = None):
        super().__init__(code=1002, message=message, status_code=401, data=data)


class AuthorizationError(APIException):
    def __init__(self, message: str = "权限不足", data: Any = None):
        super().__init__(code=1003, message=message, status_code=403, data=data)


class NotFoundError(APIException):
    def __init__(self, message: str = "资源不存在", data: Any = None):
        super().__init__(code=1004, message=message, status_code=404, data=data)


class ServerError(APIException):
    def __init__(self, message: str = "服务器内部错误", data: Any = None):
        super().__init__(code=1005, message=message, status_code=500, data=data)
