from sentence_transformers import SentenceTransformer


def load_embedding_model():
    """
    Loads the Sentence Transformer model.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def generate_embeddings(chunks, model):
    """
    Generates embeddings for each chunk.

    Args:
        chunks (list): List of chunk dictionaries.
        model: Loaded SentenceTransformer model.

    Returns:
        list: Updated chunks with embeddings.
    """

    for chunk in chunks:

        embedding = model.encode(chunk["text"])

        chunk["embedding"] = embedding

    return chunks