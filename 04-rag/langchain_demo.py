"""
LangChain RAG Demo — 用 LangChain 框架实现同样的知识库问答。

=== LangChain 的设计哲学 ===

"Build LLM applications by composing components" — 组件化、可组合。
LangChain 把一切抽象成可拼装的 Chain 和 Runnable。

核心理念：
  每个功能是一个组件 → 用 LCEL（LangChain Expression Language）串起来 → 形成 Chain

=== LangChain 的核心对象 ===

1. Document Loader — 负责读取文件（TextLoader, PyPDFLoader...）
2. Text Splitter — 负责切分文档（RecursiveCharacterTextSplitter）
3. Embeddings — 负责向量化（OpenAIEmbeddings, HuggingFaceEmbeddings...）
4. Vector Store — 向量库（Chroma, FAISS, Milvus...）
5. Retriever — 检索器（vector_store.as_retriever()）
6. Chain — 把上面的组件 + LLM 串成一条流水线

=== 和 LlamaIndex 的本质区别 ===

LangChain 是"乐高积木"：每个组件独立，你可以自由组合。
LlamaIndex 是"一体机"：从 Document 到 QueryEngine 封装好一条龙。

两者可以混用——LangChain 的 VectorStore 可以给 LlamaIndex 用，反之亦然。

=== 和手写 RAG 的对比 ===

手写版                     LangChain 对应的封装
─────────────────────────────────────────────────
load_file()               TextLoader
chunk_text()              RecursiveCharacterTextSplitter
embed_texts()             HuggingFaceEmbeddings
VectorStore.search()      vectorstore.as_retriever() + invoke()
rag_query()               RunnablePassthrough 构建的 Chain
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# --- 必须在创建 HuggingFace 模型之前设置镜像 ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from dotenv import load_dotenv
load_dotenv()

# --- LangChain 服务提供方配置 ---
# LangChain 1.0+ 推荐用 ChatModel 而不是旧的 LLM
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY", "your-deepseek-key"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    temperature=0.3,
)

# --- 本地 Embedding ---
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
)


def langchain_rag_simple():
    """方式一：逐步构建 RAG（适合理解每个组件）。

    这个写法把每一步都展开，让你看到 LangChain 的每个组件在做什么。
    对应手写版 rag_engine.py 的每一步。
    """
    from langchain_community.document_loaders import TextLoader, DirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_classic.chains import RetrievalQA

    # --- 第 1 步：加载文档 ---
    # DirectoryLoader 递归加载目录下指定类型文件
    print("第 1 步：加载文档...")
    loader = DirectoryLoader(
        r"D:\实验",
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    documents = loader.load()
    print(f"  加载了 {len(documents)} 个文档")

    # --- 第 2 步：分块 ---
    # RecursiveCharacterTextSplitter：递归切分，先按段落→句子→字符
    # chunk_size=500, chunk_overlap=80，和手写版参数一致
    print("第 2 步：文本分块...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", ".", " ", ""],  # 切分优先级
    )
    chunks = text_splitter.split_documents(documents)
    print(f"  切分成 {len(chunks)} 个文本块")

    # --- 第 3+4 步：向量化 + 存入 Chroma ---
    # Chroma.from_documents() 内部做：embedding → 存库
    print("第 3+4 步：向量化并存入 Chroma...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",  # 持久化路径
    )
    print(f"  向量库包含 {vectorstore._collection.count()} 条记录")

    # --- 第 5 步：构建问答链 ---
    # RetrievalQA 是 LangChain 预置的 RAG 链
    # 它内部做：检索 → 拼接上下文 → 发给 LLM → 返回答案
    print("第 5 步：创建问答链...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" = 把所有检索到的文档一次塞给 LLM
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,  # 返回引用来源
    )

    # --- 提问 ---
    questions = [
        "AI应用开发需要学哪些核心内容？",
        "求职的时候应该准备什么项目？",
        "Python基础学习路线里讲了什么？",
    ]

    for q in questions:
        print(f"\n{'=' * 60}")
        print(f"问题: {q}")
        print("=" * 60)
        result = qa_chain.invoke({"query": q})
        print(f"\n{result['result']}\n")
        print("参考来源：")
        for doc in result["source_documents"]:
            src = doc.metadata.get("source", "?")
            print(f"  [{src}] {doc.page_content[:80]}...")
        print()


def langchain_rag_lcel():
    """方式二：用 LCEL（LangChain Expression Language）构建。

    LCEL 使用 | 管道运算符把组件串联，是 LangChain 的推荐写法。
    """
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    # 1. 加载 + 分块
    print("加载并索引文档...")
    loader = DirectoryLoader(r"D:\实验", glob="*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80).split_documents(docs)
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db_lcel")

    # 2. 构建 LCEL 链
    # | 管道符：左边输出是右边输入，和 Unix 管道一样的思路
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 提示词模板：{context} 和 {question} 是占位符
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个知识库问答助手。根据文档内容回答问题。如果文档中没有相关信息，如实告知。回答时引用来源。"),
        ("user", "文档内容：\n{context}\n\n问题：{question}"),
    ])

    # LCEL 链：| 读作"然后"
    # 1. 接收 {"question": "..."}
    # 2. 同时把 question 传给 retriever 检索（assign 分支）
    # 3. 把 question + context 填入 prompt 模板
    # 4. 发给 LLM
    # 5. 解析成纯字符串
    chain = (
        {"context": retriever | (lambda docs: "\n\n---\n\n".join(d.page_content for d in docs)), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 3. 提问
    questions = [
        "RAG是什么，大概要学多久？",
        "求职的时候应该准备什么项目？",
    ]

    for q in questions:
        print(f"\n{'=' * 60}")
        print(f"问题: {q}")
        print("=" * 60)
        answer = chain.invoke(q)
        print(f"\n{answer}\n")


if __name__ == "__main__":
    if not os.getenv("DEEPSEEK_API_KEY", "").startswith("sk-"):
        print("请先配置 .env 中的 DEEPSEEK_API_KEY")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("方式一：逐步构建（RetrievalQA 链）")
    print("=" * 60)
    langchain_rag_simple()

    print("\n\n" + "=" * 60)
    print("方式二：LCEL 管道式构建")
    print("=" * 60)
    langchain_rag_lcel()
