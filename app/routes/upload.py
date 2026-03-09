from fastapi import APIRouter, UploadFile, File
from app.services.chunking import chunk_text
from app.services.embeddings import embed_chunks
from app.services.vector_store import store_embeddings

router = APIRouter(prefix="/upload", tags=["Upload"])

@router.post("/")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")

    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)

    store_embeddings(chunks, embeddings)

    return {"message": "Document indexed successfully"}
