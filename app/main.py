"""
Main FastAPI application for Document Intelligence RAG API.

This API provides document upload and intelligent question-answering capabilities
using RAG (Retrieval-Augmented Generation) architecture.
"""

from fastapi import FastAPI
from app.routes import upload, query
from app.services.vector_store import load_index

# Initialize FastAPI application
app = FastAPI(title="Document Intelligence RAG API")


@app.on_event("startup")
def startup_event():
    """
    Load the FAISS vector index and metadata on application startup.
    This ensures existing document embeddings are available immediately.
    """
    load_index()


# Register route handlers
app.include_router(upload.router)  # Document upload endpoint
app.include_router(query.router)   # Question-answering endpoint
