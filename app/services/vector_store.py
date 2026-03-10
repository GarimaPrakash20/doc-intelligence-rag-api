"""
Vector store service using FAISS for efficient similarity search.

Manages document embeddings storage, retrieval, and persistence using
FAISS (Facebook AI Similarity Search) for fast nearest neighbor search.
"""

import faiss
import numpy as np
import pickle
import os

# File paths for persistent storage
INDEX_PATH = "data/index.faiss"  # FAISS index containing vector embeddings
META_PATH = "data/metadata.pkl"  # Metadata (text chunks and document names)

# Embedding dimension (must match embedder model: all-MiniLM-L6-v2 = 384)
dimension = 384

# Global variables for in-memory index and metadata
index = None
metadata = []


def load_index():
    """
    Load the FAISS index and metadata from disk on application startup.
    Creates a new empty index if no existing data is found.
    """
    global index, metadata

    if os.path.exists(INDEX_PATH):
        # Load existing FAISS index from disk
        index = faiss.read_index(INDEX_PATH)
    else:
        # Create new L2 (Euclidean distance) index
        # IndexFlatL2 performs exact search (no approximation)
        index = faiss.IndexFlatL2(dimension)

    if os.path.exists(META_PATH):
        # Load metadata (text chunks and document names)
        with open(META_PATH, "rb") as f:
            metadata = pickle.load(f)


def save_index():
    """
    Persist the FAISS index and metadata to disk.
    Called after any modification to ensure data is saved.
    """
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, INDEX_PATH)

    # Save metadata as pickle file
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)


def remove_document(filename):
    """
    Remove all chunks for a specific document and rebuild the index.

    This is necessary when re-uploading a document to avoid duplicates.
    Since FAISS doesn't support deletion, we rebuild the entire index
    with only the remaining documents.

    Args:
        filename: Name of the document to remove
    """
    global index, metadata

    # Filter out metadata entries for the specified document
    remaining_metadata = [m for m in metadata if m["document"] != filename]

    if len(remaining_metadata) == len(metadata):
        # Document not found in the index, nothing to remove
        return

    # Update metadata to only include remaining documents
    metadata = remaining_metadata

    if metadata:
        # Re-create embeddings for all remaining document chunks
        from app.services.embedder import embed_documents
        remaining_texts = [m["text"] for m in metadata]
        embeddings = embed_documents(remaining_texts)

        # Create new index and add all remaining embeddings
        index = faiss.IndexFlatL2(dimension)
        embeddings_array = np.array(embeddings).astype("float32")
        index.add(embeddings_array)
    else:
        # No documents left, create empty index
        index = faiss.IndexFlatL2(dimension)

    # Persist the updated index and metadata
    save_index()


def store_embeddings(chunks, embeddings, filename):
    """
    Store document chunk embeddings and metadata in the vector store.

    Args:
        chunks: List of text chunks from the document
        embeddings: Numpy array of embeddings for each chunk
        filename: Name of the source document
    """
    global metadata

    # Ensure embeddings are in the correct format for FAISS
    embeddings = np.array(embeddings).astype("float32")

    # Add embeddings to the FAISS index
    index.add(embeddings)

    # Store metadata for each chunk (text content and source document)
    for chunk in chunks:
        metadata.append({
            "text": chunk,
            "document": filename
        })

    # Save to disk
    save_index()


def retrieve_similar(query_embedding, filename=None, top_k=8):
    """
    Retrieve the most similar document chunks for a given query embedding.

    Uses FAISS to perform efficient nearest neighbor search based on
    L2 (Euclidean) distance.

    Args:
        query_embedding: Numpy array of the query's embedding vector
        filename: Optional filter to only search within a specific document
        top_k: Number of results to return (default: 8)

    Returns:
        List of dictionaries containing 'text' and 'document' for each result
    """
    # Search for nearest neighbors
    # Request more results than needed (top_k * 3) to account for filtering
    distances, indices = index.search(query_embedding, top_k * 3)

    results = []

    for idx in indices[0]:
        # Skip invalid indices (can happen with small datasets)
        if idx >= len(metadata):
            continue

        item = metadata[idx]

        # Filter by document name if specified
        if filename and item["document"] != filename:
            continue

        results.append(item)

        # Stop when we have enough results
        if len(results) == top_k:
            break

    return results
