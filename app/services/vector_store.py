import faiss
import numpy as np
import pickle
import os

INDEX_PATH = "data/index.faiss"
META_PATH = "data/metadata.pkl"

dimension = 384  # embedding size for MiniLM
index = faiss.IndexFlatL2(dimension)

metadata = []

def store_embeddings(chunks, embeddings):
    global index, metadata

    embeddings = np.array(embeddings).astype("float32")
    index.add(embeddings)

    metadata.extend(chunks)

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)


def retrieve_similar(query_embedding, top_k=3):
    global index, metadata

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in indices[0]:
        results.append(metadata[i])

    return results
