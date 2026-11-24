
# Fixed Size Chunking Strategy

def split_into_chunks(texts: list[str], chunk_size: int = 1000) -> list:
    """
    Splits the input text into fixed-size chunks.

    Args:
        texts (list[str]): The input texts to be split.
        chunk_size (int): The size of each chunk. Default is 1000 characters.

    Returns:
        list: A list of text chunks.
    """

    if chunk_size <= 0:
        raise ValueError("Chunk size must be a positive integer.")

    overlap = chunk_size // 10  # 10% overlap
    chunks = []
    step = chunk_size - overlap
    for text in texts:
        for i in range(0, len(text), step):
            chunks.append(text[i:i + chunk_size])

    return chunks
