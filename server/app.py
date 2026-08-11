"""FastAPI entry point for RAGChef.

Thin HTTP layer on top of rag.SimpleRAG: builds the RAG instance once at
startup, exposes POST /ask for the Chrome extension to call, and GET / as a
basic health check (used by render.yaml's healthCheckPath).
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RECIPES_PATH = os.path.join(os.path.dirname(__file__), "data", "recipes")


# Build the RAG instance once, at import time, rather than lazily on first
# request. RAGConfigError is intentionally allowed to propagate: a
# misconfigured deployment (e.g. missing API key) should fail fast at boot,
# not come up healthy and then 500 on every request.
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
    return {"message": "RAGChef backend is running"}
