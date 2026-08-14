import numpy as np


def search(query, model, index, chunks, top_k=3, distance_threshold=1.0):
    """
    Search the FAISS index and determine whether the retrieved
    results are relevant enough to use.

    Args:
        query (str): User's natural-language query.
        model: Loaded Sentence Transformer model.
        index: FAISS vector index.
        chunks (list): Document chunks.
        top_k (int): Number of results to retrieve.
        distance_threshold (float): Maximum allowed L2 distance.

    Returns:
        dict: Retrieval status, best distance, and retrieved results.
    """

    # Convert query into an embedding
    query_embedding = model.encode([query])

    # Convert embedding to NumPy float32 for FAISS
    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # Search FAISS
    distances, indices = index.search(
        query_embedding,
        top_k
    )

    # Best result = smallest distance
    best_distance = float(distances[0][0])

    # Build retrieved results
    results = []

    for rank, (distance, index_id) in enumerate(
        zip(distances[0], indices[0]),
        start=1
    ):

        if index_id == -1:
            continue

        chunk = chunks[index_id].copy()

        chunk["rank"] = rank
        chunk["distance"] = float(distance)
        chunk["faiss_index"] = int(index_id)

        results.append(chunk)

    # Check whether the query is relevant enough
    accepted = best_distance <= distance_threshold

    return {
        "accepted": accepted,
        "best_distance": best_distance,
        "results": results
    }