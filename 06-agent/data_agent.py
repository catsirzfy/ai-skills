"""
数据分析 Agent — 求职作品级项目。

=== 功能 ===

用户用自然语言描述分析需求，Agent 自动：
1. 读取数据文件（CSV/Excel）
2. 编写并执行 Python 分析代码
3. 生成图表
4. 输出分析报告

=== 架构 ===

用户 → Agent（LangChain AgentExecutor）
         ├── load_data       # 读取 CSV/Excel
         ├── describe_data   # 数据概览（行数、列名、类型、缺失值）
         ├── analyze_python  # 执行分析代码
         └── save_plot       # 保存图表

=== 运行方式 ===

    # 准备测试数据
    python data_agent.py --create-sample

    # 交互式分析
    python data_agent.py
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ================================================================
# 工具 1：加载数据
# ================================================================
def load_data(filepath: str) -> str:
    """加载 CSV 或 Excel 文件，返回数据概览。

    这是分析的第一步——Agent 需要先知道数据长什么样。
    """
    import pandas as pd

    path = Path(filepath)
    if not path.exists():
        # 尝试在 data 目录找
        path = DATA_DIR / filepath
        if not path.exists():
            return f"错误：文件不存在 {filepath}"

    try:
        if path.suffix == ".csv":
            df = pd.read_csv(path)
        elif path.suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            return f"不支持的文件格式: {path.suffix}"

        # 返回数据结构信息（不返回完整数据，太长）
        info = {
            "文件名": str(path.name),
            "行数": len(df),
            "列数": len(df.columns),
            "列名": list(df.columns),
            "数据类型": {c: str(t) for c, t in df.dtypes.items()},
            "缺失值": {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().sum() > 0},
            "前5行": df.head().to_dict(orient="records"),
            "数值列统计": df.describe().to_dict() if len(df.select_dtypes(include="number").columns) > 0 else {},
        }

        # 存到全局变量供后续分析使用
        global _current_df
        _current_df = df

        return json.dumps(info, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return f"加载失败: {e}"


_current_df = None  # 当前加载的数据框


# ================================================================
# 工具 2：数据概览
# ================================================================
def describe_data() -> str:
    """获取当前数据集的详细统计信息。"""
    global _current_df
    if _current_df is None:
        return "请先使用 load_data 加载数据文件。"

    df = _current_df
    info = {
        "形状": f"{df.shape[0]} 行 × {df.shape[1]} 列",
        "列名及类型": {c: str(t) for c, t in df.dtypes.items()},
        "缺失值总数": int(df.isna().sum().sum()),
        "数值列统计": {},
        "分类列": {},
    }

    # 数值列统计
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        info["数值列统计"] = df[num_cols].describe().to_dict()

    # 分类列信息
    cat_cols = df.select_dtypes(include="object").columns
    for c in cat_cols:
        unique = df[c].nunique()
        if unique < 20:  # 只展示类别少的
            info["分类列"][c] = {
                "唯一值数量": unique,
                "最常见": df[c].value_counts().head(5).to_dict(),
            }

    return json.dumps(info, ensure_ascii=False, indent=2, default=str)


# ================================================================
# 工具 3：执行分析代码
# ================================================================
def analyze_with_python(code: str) -> str:
    """用 Python 分析当前数据。可以访问 df 变量（已加载的数据）。

    df 是一个 pandas DataFrame，你可以用它做任何分析。
    输出用 print() 显示。

    示例：
        print(df.groupby('月份')['销售额'].sum())
        print(df['年龄'].describe())
        print(df.corr())
    """
    global _current_df
    if _current_df is None:
        return "错误：请先使用 load_data 加载数据。当前没有数据集。"

    import io
    import matplotlib
    matplotlib.use("Agg")  # 非交互模式
    import matplotlib.pyplot as plt

    try:
        buf = io.StringIO()
        _stdout = sys.stdout
        sys.stdout = buf

        local_vars = {"df": _current_df, "pd": __import__("pandas"), "np": __import__("numpy"), "plt": plt}

        exec(code, {"__builtins__": {
            "print": print, "range": range, "len": len, "sum": sum, "max": max, "min": min,
            "sorted": sorted, "list": list, "dict": dict, "set": set, "str": str,
            "int": int, "float": float, "bool": bool, "abs": abs, "round": round,
            "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
            "isinstance": isinstance, "type": type, "True": True, "False": False,
            "None": None, "Exception": Exception, "ValueError": ValueError,
        }}, local_vars)

        sys.stdout = _stdout
        output = buf.getvalue().strip()
        return output if output else "(代码执行完成，无输出)"
    except Exception as e:
        sys.stdout = _stdout
        return f"错误: {e}"


# ================================================================
# 工具 4：生成图表
# ================================================================
def save_plot(code: str, filename: str = "plot.png") -> str:
    """生成图表并保存为 PNG 文件。

    可以用 plt 和 df 变量。示例：
        plt.figure()
        df.groupby('月份')['销售额'].sum().plot(kind='bar')
        plt.title('月度销售额')
        plt.tight_layout()
    """
    global _current_df
    if _current_df is None:
        return "错误：请先使用 load_data 加载数据。"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    save_path = DATA_DIR / filename

    try:
        local_vars = {"df": _current_df, "plt": plt, "np": np, "pd": __import__("pandas")}
        exec(code, {"__builtins__": {"print": print, "range": range, "len": len, "sum": sum, "max": max, "min": min, "sorted": sorted, "list": list, "dict": dict, "set": set}}, local_vars)

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        return f"图表已保存: {save_path}"
    except Exception as e:
        plt.close("all")
        return f"错误: {e}"


# ================================================================
# 构建 LangChain Agent
# ================================================================
from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate


@tool
def lc_load_data(filepath: str) -> str:
    """加载 CSV 或 Excel 数据文件。这是数据分析的第一步。参数是文件路径。"""
    return load_data(filepath)


@tool
def lc_describe_data() -> str:
    """获取当前数据集的详细统计信息（行数、列名、类型、缺失值、统计概要）。"""
    return describe_data()


@tool
def lc_analyze_python(code: str) -> str:
    """用 Python 代码分析当前数据。可以使用 df 变量（pandas DataFrame）。用 print() 输出结果。"""
    return analyze_with_python(code)


@tool
def lc_save_plot(code: str, filename: str = "plot.png") -> str:
    """生成并保存图表。用 plt 和 df 绘图。参数：code=绘图代码, filename=保存的文件名。"""
    return save_plot(code, filename)


AGENT_TOOLS = [lc_load_data, lc_describe_data, lc_analyze_python, lc_save_plot]


def build_data_agent():
    """构建数据分析 Agent。"""
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY", "your-deepseek-key"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0.2,  # 数据分析需要精确，用低温
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "你是一个数据分析专家 Agent。你能加载数据、分析数据、生成图表。\n"
            "分析流程：\n"
            "1. 先用 load_data 加载数据文件\n"
            "2. 用 describe_data 查看数据概览\n"
            "3. 根据用户需求，用 analyze_python 写分析代码\n"
            "4. 如果需要可视化，用 save_plot 生成图表\n"
            "5. 最后用文字总结分析结果\n"
            "使用中文回答。"
        )),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, AGENT_TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=AGENT_TOOLS, verbose=True, max_iterations=8, handle_parsing_errors=True)


# ================================================================
# 创建示例数据
# ================================================================
def create_sample_data():
    """生成示例销售数据文件。"""
    import pandas as pd
    import numpy as np

    np.random.seed(42)

    months = pd.date_range("2024-01-01", periods=24, freq="ME")
    products = ["笔记本电脑", "手机", "平板", "耳机", "键盘"]

    data = []
    for month in months:
        for product in products:
            base_sales = np.random.randint(50, 200)
            seasonal = 1.5 if month.month in [6, 7, 11, 12] else 1.0
            data.append({
                "月份": month.strftime("%Y-%m"),
                "产品": product,
                "销量": int(base_sales * seasonal + np.random.randint(-20, 20)),
                "单价": np.random.choice([199, 299, 499, 999, 1999]),
                "地区": np.random.choice(["华北", "华东", "华南", "西南", "西北"]),
            })

    df = pd.DataFrame(data)
    df["销售额"] = df["销量"] * df["单价"]

    path = DATA_DIR / "sales_data.csv"
    df.to_csv(path, index=False)
    print(f"示例数据已创建: {path}")
    print(f"  {len(df)} 行, {len(df.columns)} 列: {list(df.columns)}")
    print(f"  前 3 行:\n{df.head(3)}")
    return path


if __name__ == "__main__":
    import sys

    if "--create-sample" in sys.argv:
        create_sample_data()
        sys.exit(0)

    # 确保数据存在
    if not list(DATA_DIR.glob("*.csv")):
        print("未找到数据文件，正在创建示例数据...")
        create_sample_data()
        print()

    agent = build_data_agent()

    print("=" * 60)
    print("数据分析 Agent")
    print("=" * 60)
    print(f"数据目录: {DATA_DIR}")
    print(f"可用文件: {[f.name for f in DATA_DIR.glob('*')]}")
    print("输入 /quit 退出\n")

    while True:
        try:
            q = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if q.lower() in ("/quit", "/q", "/exit"):
            break
        if not q:
            continue

        result = agent.invoke({"input": q})
        print(f"\nAgent: {result['output']}\n")
