"""路由注册中心 — 所有路由集中管理，避免 main.py 臃肿。

新增 API 模块只需：
1. 在 app/api/client/v1/ 或 backoffice/v1/ 下创建文件
2. 在 CLIENT_ROUTES 或 BACKOFFICE_ROUTES 中添加一行配置
"""
from typing import List
from dataclasses import dataclass
from app.core.config import settings


@dataclass
class RouteConfig:
    module_path: str   # 如 "app.api.client.v1.demo"
    prefix: str        # 如 "/api/v1/demo"
    tags: List[str]    # 如 ["demo"]


# ---- 客户端路由 ----
CLIENT_ROUTES = [
    RouteConfig("app.api.client.v1.demo", f"{settings.API_V1_STR}/demo", ["Demo"]),
    # 在这里添加新的客户端路由...
]

# ---- 后台路由 ----
BACKOFFICE_ROUTES = [
    RouteConfig("app.api.backoffice.v1.demo", f"{settings.API_V1_STR}/backoffice/demo", ["Backoffice Demo"]),
    # 在这里添加新的后台路由...
]


def register_routes(app, routes: List[RouteConfig]):
    """动态注册路由。

    每个 RouteConfig 指向一个 Python 模块，模块中必须有 router 对象。
    """
    for config in routes:
        *package, module_name = config.module_path.split(".")
        module = __import__(config.module_path, fromlist=[module_name])
        app.include_router(module.router, prefix=config.prefix, tags=config.tags)
