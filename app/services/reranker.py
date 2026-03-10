"""
Reranking service using a cross-encoder model for improved relevance scoring.

The reranker refines initial retrieval results by computing relevance scores
between the query and each candidate document using a more sophisticated model.
"""

from sentence_transformers import CrossEncoder

# Load cross-encoder reranker model
# This model provides more accurate relevance scoring than simple cosine similarity
reranker = CrossEncoder("BAAI/bge-reranker-base")


def rerank(query, docs, top_k=3):
    """
    Rerank documents based on their relevance to the query.

    Uses a cross-encoder to compute precise relevance scores by jointly
    encoding the query and each document together.

    Args:
        query: User's question string
        docs: List of document text chunks to rerank
        top_k: Number of top documents to return (default: 3)

    Returns:
        List of top-k most relevant document texts, sorted by relevance score
    """
    # Create query-document pairs for the cross-encoder
    pairs = [(query, doc) for doc in docs]

    # Compute relevance scores for each pair
    scores = reranker.predict(pairs)

    # Combine documents with their scores
    scored_docs = list(zip(docs, scores))

    # Sort by score in descending order (highest relevance first)
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Return only the top-k document texts (without scores)
    return [doc for doc, score in scored_docs[:top_k]]
