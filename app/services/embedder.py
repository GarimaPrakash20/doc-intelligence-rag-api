"""
Embedding service using sentence-transformers for document and query vectorization.

Uses all-MiniLM-L6-v2 model which produces 384-dimensional embeddings.
This model is optimized for semantic similarity tasks.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# Load the embedding model (384-dimensional)
# This model is lightweight and fast, suitable for production use
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_documents(chunks):
    """
    Convert document chunks into dense vector embeddings.

    Args:
        chunks: List of text strings (document chunks)

    Returns:
        numpy array of shape (n_chunks, 384) containing embeddings
    """
    embeddings = model.encode(chunks)
    return np.array(embeddings).astype("float32")


def embed_query(query):
    """
    Convert a user query into a dense vector embedding.
    Uses the same model as document embeddings for consistency.

    Args:
        query: String containing the user's question

    Returns:
        numpy array of shape (1, 384) containing the query embedding
    """
    embedding = model.encode([query])
    return np.array(embedding).astype("float32")
