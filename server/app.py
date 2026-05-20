from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import SimpleRAG

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = SimpleRAG("data/recipes.md")

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    answer = rag.ask(request.question)
    return {"answer": answer}

@app.get("/")
def root():
    return {"message": "RAGChef backend is running"}