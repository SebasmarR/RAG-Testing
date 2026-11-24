import chromadb

chroma_client = chromadb.PersistentClient(path="./rag_chroma")


def get_or_create_collection(collection_name="rag_collection", metadata=None):
    """
    Retrieve an existing collection from the ChromaDB client.

    Args:
        collection_name (str): The name of the collection to retrieve.

    Returns:
        chromadb.api.models.Collection: The retrieved collection object.
    """
    if metadata is None:
        metadata = {"description": "RAG collection for storing embeddings."}

    collection = chroma_client.get_or_create_collection(
        name=collection_name, metadata=metadata
    )

    return collection


def store_embeddings(collection: str = "rag_collection", chunks=None, embeddings=None):
    """
    Store embeddings in the specified ChromaDB collection.

    Args:
        collection (chromadb.api.models.Collection): The collection to store embeddings in.
        chunks (list[str]): A list of text chunks corresponding to the embeddings.
        embeddings (np.ndarray): A 2D array of embeddings to store.

    Returns:
        None
    """

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            "The number of chunks must match the number of embeddings.")

    chunks_ids = list(map("chunk_{}".format, range(len(chunks))))

    collection.add(
        ids=chunks_ids,
        documents=chunks,
        embeddings=embeddings.tolist()
    )
