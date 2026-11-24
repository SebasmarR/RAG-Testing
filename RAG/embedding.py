import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')


def create_embeddings(chunks: list[str]) -> np.ndarray:
    """
    Creates embeddings for a list of text chunks using the
    SentenceTransformer model.

    Args: 
        chunks (list[str]): A list of text chunks to be embedded.

    Returns:
        np.ndarray: A 2D array where each row corresponds to the embedding of a chunk.
    """

    if not chunks:
        raise ValueError("The list of chunks is empty.")

    embeddings = model.encode(chunks, convert_to_numpy=True,
                              normalize_embeddings=True)
    return embeddings
