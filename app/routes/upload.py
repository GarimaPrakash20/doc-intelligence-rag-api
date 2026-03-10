"""
Document upload route for processing and indexing documents.

Supports PDF and TXT files. Extracts text, cleans it, chunks into
manageable pieces, and stores embeddings in the vector store.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import re
from pypdf import PdfReader

from app.services.embedder import embed_documents
from app.services.vector_store import store_embeddings, remove_document

router = APIRouter()


def clean_text(text):
    """
    Clean extracted text while preserving sentence structure.

    Normalizes bullet points, whitespace, and formatting artifacts
    that commonly appear in PDF extraction.

    Args:
        text: Raw extracted text from document

    Returns:
        Cleaned text string
    """
    # Replace bullet points and special chars with simple dashes
    # Handles various Unicode bullet characters
    text = re.sub(r'[●○■□▪▫•◦]', '- ', text)

    # Normalize whitespace while preserving paragraph breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Keep double newlines (paragraphs)
    text = re.sub(r'\n', ' ', text)  # Convert single newlines to spaces
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces to single space

    return text.strip()


def split_into_chunks(text, chunk_size=600, overlap=100):
    """
    Split text into overlapping chunks on sentence boundaries.

    Intelligent chunking that:
    - Respects sentence boundaries (doesn't cut mid-sentence)
    - Maintains context overlap between chunks
    - Keeps chunks within a target size

    Args:
        text: Cleaned document text
        chunk_size: Target maximum characters per chunk (default: 600)
        overlap: Number of characters to overlap between chunks (default: 100)

    Returns:
        List of text chunks
    """
    # Split into sentences using common sentence terminators
    # Lookahead assertion ensures we split AFTER punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_size = len(sentence)

        # If adding this sentence exceeds chunk_size and we have content
        if current_size + sentence_size > chunk_size and current_chunk:
            # Save the current chunk
            chunks.append(' '.join(current_chunk))

            # Prepare overlap for next chunk
            overlap_text = ' '.join(current_chunk)
            if len(overlap_text) > overlap:
                # Keep last few sentences for context continuity
                current_chunk = [s for s in current_chunk if len(' '.join(current_chunk[current_chunk.index(s):])) <= overlap]
                current_size = len(' '.join(current_chunk))
            else:
                # Start fresh if current chunk is smaller than overlap
                current_chunk = []
                current_size = 0

        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_size += sentence_size + 1  # +1 for space between sentences

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def extract_text_from_txt(file_path):
    """
    Extract text from a plain text file.

    Args:
        file_path: Path to the .txt file

    Returns:
        Extracted text content
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file using pypdf.

    Iterates through all pages and concatenates the extracted text.

    Args:
        file_path: Path to the .pdf file

    Returns:
        Extracted text content from all pages
    """
    reader = PdfReader(file_path)

    text = ""

    # Extract text from each page
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    return text


@router.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document for RAG system.

    Workflow:
    1. Save uploaded file
    2. Extract text based on file type
    3. Clean and normalize text
    4. Split into overlapping chunks
    5. Generate embeddings
    6. Remove old version (if re-uploading)
    7. Store in vector database

    Args:
        file: Uploaded file (PDF or TXT)

    Returns:
        Success message with processing statistics
    """
    # Create uploads directory if it doesn't exist
    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)

    file_path = os.path.join(upload_folder, file.filename)

    # Save uploaded file to disk
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Extract and clean text based on file type
    if file.filename.endswith(".txt"):
        text = extract_text_from_txt(file_path)
        text = clean_text(text)

    elif file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
        text = clean_text(text)

    else:
        raise HTTPException(
            status_code=400,
            detail="Only TXT and PDF files are supported"
        )

    # Validate that we extracted some text
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Document contains no readable text"
        )

    # Split text into manageable chunks
    chunks = split_into_chunks(text)

    # Generate vector embeddings for all chunks
    embeddings = embed_documents(chunks)

    # Remove old chunks if this document was previously uploaded
    # This prevents duplicate chunks in the vector store
    remove_document(file.filename)

    # Store new embeddings and metadata
    store_embeddings(chunks, embeddings, file.filename)

    return {
        "message": "Document processed successfully",
        "filename": file.filename,
        "chunks_created": len(chunks)
    }
