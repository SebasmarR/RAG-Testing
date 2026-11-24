import chromadb
from .storing import get_or_create_collection
from .embedding import create_embeddings

chroma_client = chromadb.PersistentClient(path="./rag_chroma")


def query_search(query="", collection: str = "rag_collection", top_k: int = 5):
    """
    Perform a similarity search in the specified ChromaDB collection.

    Args:
        collection (chromadb.api.models.Collection): The collection to search in.
        query (str): The query string to search for.
        top_k (int): The number of top results to return. Default is 5.

    Returns:
        list: A list of the top_k most similar documents to the query.
    """

    collection = get_or_create_collection(collection)

    # Create embeddings for the query
    query_embedding = create_embeddings([query])

    # Perform the similarity search
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k
    )

    return results['documents']
