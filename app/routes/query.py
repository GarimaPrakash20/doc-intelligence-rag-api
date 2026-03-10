"""
Query route for answering questions using RAG (Retrieval-Augmented Generation).

Implements a multi-stage retrieval pipeline:
1. Embed query
2. Retrieve candidate documents (vector similarity)
3. Rerank for relevance (cross-encoder)
4. Generate answer (LLM)
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from app.services.embedder import embed_query
from app.services.vector_store import retrieve_similar
from app.services.reranker import rerank
from app.services.llm import generate_answer

router = APIRouter()


class QueryRequest(BaseModel):
    """Request model for question-answering queries."""
    question: str  # The user's question
    document: Optional[str] = None  # Optional: filter to specific document


@router.post("/query/")
async def query_docs(request: QueryRequest):
    """
    Answer a question using the RAG pipeline.

    Pipeline stages:
    1. Query Embedding: Convert question to vector representation
    2. Initial Retrieval: Find top-k similar chunks using FAISS (fast)
    3. Deduplication: Remove duplicate text chunks
    4. Reranking: Refine results using cross-encoder (accurate)
    5. Answer Generation: Use LLM to synthesize answer from top results

    Args:
        request: QueryRequest containing question and optional document filter

    Returns:
        Dictionary with:
        - question: Original question
        - document: Document name (if filtered)
        - answer: Generated answer
        - sources: Top 3 relevant text chunks with document references
    """
    # Step 1: Convert question to embedding vector
    query_embedding = embed_query(request.question)

    # Step 2: Retrieve top 8 similar chunks from vector store
    # Using more than needed (8 vs final 3) to account for deduplication
    docs = retrieve_similar(
        query_embedding,
        filename=request.document,  # Optional filter by document
        top_k=8
    )

    # Step 3: Deduplicate chunks while preserving order
    # dict.fromkeys() maintains insertion order and removes duplicates
    texts = list(dict.fromkeys([d["text"] for d in docs]))

    # Step 4: Rerank using cross-encoder for better relevance
    # This is slower but more accurate than cosine similarity
    reranked_docs = rerank(request.question, texts, top_k=3)

    # Step 5: Generate answer using LLM based on top reranked chunks
    answer = generate_answer(request.question, reranked_docs)

    # Prepare source references for the response
    # Map reranked texts back to their document names
    text_to_doc = {d["text"]: d["document"] for d in docs}
    sources = [
        {"text": text, "document": text_to_doc.get(text, request.document)}
        for text in reranked_docs
    ]

    return {
        "question": request.question,
        "document": request.document,
        "answer": answer,
        "sources": sources  # Return actual reranked sources (not original docs)
    }
