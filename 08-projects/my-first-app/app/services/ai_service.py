"""AI 调用 Service — 封装 DeepSeek API。"""
from openai import AsyncOpenAI
from app.core.config import settings

client = AsyncOpenAI(
    api_key=settings.DEEPSEEK_API_KEY,
    base_url=settings.DEEPSEEK_BASE_URL,
)


async def ask_ai(question: str) -> tuple[str, str, int]:
    """调用 AI 回答问题。

    返回：(answer, model_name, tokens_used)
    """
    response = await client.chat.completions.create(
        model=settings.DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": question}],
        temperature=0.7,
        max_tokens=2000,
    )
    answer = response.choices[0].message.content
    model = response.model
    tokens = response.usage.total_tokens
    return answer, model, tokens
