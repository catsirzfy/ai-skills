"""FastAPI 应用工厂函数。

create_app() 创建 app 实例，注册路由和异常处理。
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.route.router_registry import CLIENT_ROUTES, BACKOFFICE_ROUTES, register_routes
from app.exceptions import APIException
from app.db.base import close_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动/关闭时的操作。"""
    # 启动时
    yield
    # 关闭时
    await close_db()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        docs_url="/docs" if settings.ENV == "development" else None,
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.FRONTEND_URL],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    register_routes(app, CLIENT_ROUTES)
    register_routes(app, BACKOFFICE_ROUTES)

    # 全局异常处理
    @app.exception_handler(APIException)
    async def api_exception_handler(request: Request, exc: APIException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.detail, "data": exc.data},
        )

    return app
