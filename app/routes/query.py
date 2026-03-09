from fastapi import APIRouter
from pydantic import BaseModel
from app.services.embeddings import model
from app.services.vector_store import retrieve_similar
from app.services.llm import generate_answer
import numpy as np

router = APIRouter(prefix="/query", tags=["Query"])


class QueryRequest(BaseModel):
    question: str


@router.post("/")
async def query_docs(request: QueryRequest):

    query_embedding = model.encode([request.question])
    query_embedding = np.array(query_embedding).astype("float32")

    docs = retrieve_similar(query_embedding)

    answer = generate_answer(request.question, docs)

    return {
        "question": request.question,
        "answer": answer,
        "context_used": docs
    }
