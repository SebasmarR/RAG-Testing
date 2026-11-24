from importlib_resources import files

from .loader import load_folder
from .chunking import split_into_chunks
from .embedding import create_embeddings
from .storing import get_or_create_collection, store_embeddings
from .search import query_search


def rag_preloading(folder_path: str = "documents", chunk_size: int = 1000) -> None:
    """
    Preload all files from the 'documents' folder, split them into chunks,
    create embeddings for each chunk, and store them in a collection.

    Returns:
        None
    """
    folder_path = "documents"
    files = load_folder(folder_path)  # List of txts

    chunks = split_into_chunks(files, chunk_size)  # List of chunks

    embeddings = create_embeddings(chunks)  # List of embeddings

    collection = get_or_create_collection()  # Get the collection

    # Store embeddings in the collection
    store_embeddings(collection, chunks, embeddings)
