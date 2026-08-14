import faiss
import numpy as np


def create_faiss_index(embeddings):
    """
    Create a FAISS index from embedding vectors.

    Args:
        embeddings: NumPy array containing embedding vectors.

    Returns:
        FAISS index containing the embeddings.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


def search_faiss(index, query_embedding, top_k=3):
    """
    Search the FAISS index for the nearest vectors.

    Args:
        index: FAISS index containing document embeddings.
        query_embedding: Embedding of the user's query.
        top_k (int): Number of nearest results to retrieve.

    Returns:
        distances: Distances of retrieved vectors.
        indices: Positions of retrieved vectors in the FAISS index.
    """

    query_embedding = np.array(query_embedding).astype("float32")

    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    distances, indices = index.search(query_embedding, top_k)

    return distances, indices