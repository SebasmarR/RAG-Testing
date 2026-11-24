import numpy as np
from .embedding import create_embeddings


def search_with_numpy(query: str, chunks: list[str], chunk_embeddings: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
    """
    Searches for the most relevant chunks based on the query using NumPy.

    Args:
        query (str): The search query.
        chunks (list[str]): A list of text chunks to search within.
        chunk_embeddings (np.ndarray): Precomputed embeddings for the chunks.
        top_k (int): The number of top results to return. Default is 5.

    Returns:
        list[tuple[str, float]]: A list of tuples containing the top-k chunks and their similarity scores.
    """

    # Get the embedding for the query
    query_embedding = create_embeddings([query])[0]

    # Compute similarities using dot product.
    similarities = np.dot(chunk_embeddings, query_embedding)

    # Get the indices of the top-k most similar chunks
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]

    # Prepare the results
    results = [(chunks[i], similarities[i]) for i in top_k_indices]

    return results
