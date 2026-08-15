"""FastAPI entry point for RAGChef.

Thin HTTP layer on top of rag.SimpleRAG: builds the RAG instance once at
startup, exposes POST /ask for the Chrome extension to call, and GET / as a
basic health check (used by render.yaml's healthCheckPath).

# 中文说明:
# 这是整个后端的入口文件。它本身不做任何检索/生成逻辑,只是把
# rag.py 里的 SimpleRAG 包装成 HTTP 接口,供 Chrome 插件调用。
"""

import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag import SimpleRAG, RAGConfigError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ragchef")

app = FastAPI()

# TODO(security): restrict allow_origins to the extension's origin before
# shipping to production; "*" is only acceptable for local development.
# 中文: 允许所有来源跨域访问,方便本地开发时 Chrome 插件调用;
# 上线前应改成只允许插件自己的 origin,否则任何网页都能调这个接口。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 中文: 拼出 data/recipes 目录的绝对路径,这是知识库(50 个菜谱 md 文件)所在位置
RECIPES_PATH = os.path.join(os.path.dirname(__file__), "data", "recipes")


# Build the RAG instance once, at import time, rather than lazily on first
# request. RAGConfigError is intentionally allowed to propagate: a
# misconfigured deployment (e.g. missing API key) should fail fast at boot,
# not come up healthy and then 500 on every request.
# 中文: 在模块加载时(也就是服务启动时)就构建 SimpleRAG 实例,而不是等
# 第一个请求来了才建。这样做的好处是:如果配置有问题(比如没配 API key、
# 知识库是空的),服务会在启动阶段直接崩溃退出("快速失败"),而不是
# 表面上启动成功、实际每个请求都 500,让人误以为服务是健康的。
try:
    rag = SimpleRAG(RECIPES_PATH)
except RAGConfigError as e:
    logger.error("RAGChef failed to start: %s", e)
    raise


class QueryRequest(BaseModel):
    """Request body for POST /ask.

    Attributes:
        question: The user's natural-language question. Pydantic validates
            this field automatically, so a missing/invalid value returns an
            HTTP 422 before ask_question() runs.
    """
    # 中文: 请求体的数据模型,只有一个字段 question。
    # Pydantic 会自动校验类型,字段缺失或类型不对时会在进入函数体之前
    # 就直接返回 422,不需要手写校验逻辑。

    question: str


@app.post("/ask")
def ask_question(request: QueryRequest) -> dict:
    """Answers a recipe question via the RAG pipeline.

    Args:
        request: The parsed request body containing the user's question.

    Returns:
        A dict of the form {"answer": str}.

    Raises:
        HTTPException: With status 500 if rag.ask() raises an unexpected
            exception (e.g. a bug). Expected LLM-provider failures are
            already handled inside SimpleRAG.ask() and returned as a normal
            answer string, so they never reach this handler. Internal error
            details are never included in the response.
    """
    # 中文: 非流式问答接口。核心逻辑全部委托给 rag.ask(),这里只负责
    # 兜底捕获"预料之外"的异常(比如代码 bug),转成统一的 500 错误,
    # 并且不把内部异常细节暴露给客户端。像"LLM 调用失败"这种"预料之内"
    # 的失败,SimpleRAG.ask() 内部已经处理成一句正常的回答文本了,
    # 不会走到这个 except 分支。
    try:
        answer = rag.ask(request.question)
    except Exception:
        logger.exception("Unexpected error while answering question.")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while generating the answer. Please try again.",
        )
    return {"answer": answer}


@app.post("/ask/stream")
def ask_question_stream(request: QueryRequest) -> StreamingResponse:
    """Streaming counterpart of /ask: streams the answer as plain text chunks.

    Used by the Chrome extension so the answer appears incrementally as
    DeepSeek generates it, instead of the extension staring at "Thinking..."
    for the entire generation time before anything shows up. The response
    body is plain UTF-8 text, not JSON or SSE -- there's nothing but answer
    text to send, so a bare chunked text/plain stream is the whole protocol;
    the client just needs to read and append.

    Args:
        request: The parsed request body containing the user's question.

    Returns:
        A StreamingResponse of the answer text.

    Note:
        Unlike /ask, an unexpected error here can't turn into a clean HTTP
        500 once streaming has already started -- some bytes may already be
        on the wire. rag.ask_stream() itself doesn't raise on expected
        LLM-provider failures (same as ask()), but if something unexpected
        still goes wrong mid-stream, it's caught here and appended to the
        response as a plain-text message instead of surfacing as an HTTP
        error status, since the status code has already been sent by the
        time that could happen.
    """
    # 中文: /ask 的流式版本。逐块(chunk)把 DeepSeek 生成的文本吐给前端,
    # 用户能看到答案边生成边显示,而不是等全部生成完才一次性出现。
    # 返回的是纯文本流(text/plain),不是 JSON,也不是 SSE 协议,
    # 因为这里只需要传答案文字,不需要额外的事件类型/结构化字段,
    # 客户端只要不断读取并拼接即可。
    #
    # 注意: 一旦开始流式传输,HTTP 状态码已经发送出去了,这时如果中途
    # 出错,没法再改成返回 500 了(字节已经在传输中)。所以这里的做法是
    # 把错误信息作为普通文本追加到已经在传输的内容后面,而不是像 /ask
    # 那样抛 HTTPException。
    def generate():
        try:
            for chunk in rag.ask_stream(request.question):
                yield chunk
        except Exception:
            logger.exception("Unexpected error while streaming answer.")
            yield "\n\nSomething went wrong while generating the answer. Please try again."

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@app.get("/")
def root() -> dict:
    """Liveness/health check endpoint, also used as Render's healthCheckPath.

    Returns:
        A dict of the form {"message": str}.
    """
    # 中文: 健康检查接口。Render 部署平台会定期请求这个路径,
    # 用来判断服务是否还活着(存活探针)。
    return {"message": "RAGChef backend is running"}
