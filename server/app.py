import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import SimpleRAG, RAGConfigError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ragchef")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RECIPES_PATH = os.path.join(os.path.dirname(__file__), "data", "recipes.md")

try:
    rag = SimpleRAG(RECIPES_PATH)
except RAGConfigError as e:
    # Fail fast and loud: a misconfigured server (missing API key, empty
    # knowledge base, etc.) should never silently start and 500 on every request.
    logger.error("RAGChef failed to start: %s", e)
    raise


class QueryRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        answer = rag.ask(request.question)
    except Exception:
        logger.exception("Unexpected error while answering question.")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while generating the answer. Please try again.",
        )
    return {"answer": answer}


@app.get("/")
def root():
    return {"message": "RAGChef backend is running"}
