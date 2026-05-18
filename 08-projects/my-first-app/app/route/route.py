"""FastAPI 应用工厂函数。"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.route.router_registry import CLIENT_ROUTES, BACKOFFICE_ROUTES, register_routes
from app.exceptions import APIException
from app.db.base import close_db

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_db()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        docs_url="/docs" if settings.ENV == "development" else None,
        lifespan=lifespan,
    )

    # CORS — 开发环境允许所有来源
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 前端首页 — 访问 http://localhost:8000 就能看到网页
    @app.get("/")
    async def home():
        index_path = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "前端文件不存在，请创建 static/index.html"}

    # 注册 API 路由
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
