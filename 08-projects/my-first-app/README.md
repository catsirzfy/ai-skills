# FastAPI 项目脚手架

从 AI-Interview 项目抽取的通用架构，拿来即用。

## 架构

```
app/
├── core/           # 配置(config) + 认证(security)
├── db/             # 数据库引擎 + 会话管理
├── models/         # SQLAlchemy 模型（BaseModel 基类）
├── schemas/        # Pydantic 模型 + ApiResponse
├── exceptions/     # 分层异常体系
├── api/
│   ├── client/     # 用户端 API + JWT 认证依赖
│   └── backoffice/ # 后台管理 API
├── services/       # 业务逻辑层
├── route/          # 路由注册中心
└── common/         # 公共工具
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境
cp .env.example .env
# 编辑 .env 填入数据库和 AI 配置

# 3. 启动 PostgreSQL（Docker 或本地）
docker run -d --name postgres -p 5432:5432 -e POSTGRES_USER=demo -e POSTGRES_PASSWORD=demo123 -e POSTGRES_DB=myproject postgres:16-alpine

# 4. 数据库迁移
alembic upgrade head

# 5. 启动
python main.py
# 访问 http://localhost:8000/docs
```

## 通用能力

| 模块 | 说明 |
|------|------|
| `core/config.py` | pydantic-settings 自动读 .env |
| `core/security.py` | JWT 生成/验证 + 密码哈希 |
| `exceptions/` | 5 种异常类型，统一错误码 |
| `schemas/response.py` | ApiResponse.success() / .error() |
| `db/base.py` | 异步引擎 + 连接池优化 |
| `db/session.py` | FastAPI 依赖注入 get_db() |
| `api/client/deps.py` | JWT 认证依赖 get_current_user() |
| `route/` | 路由注册中心，动态注册 |

## 新增 API 模块

1. 在 `app/api/client/v1/` 下创建 `your_module.py`
2. 在 `app/route/router_registry.py` 的 CLIENT_ROUTES 中添加一行
3. 在 `app/services/` 下创建对应的 Service
