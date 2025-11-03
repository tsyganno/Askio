def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 100):
    """
    Простая функция для разбиения текста на перекрывающиеся чанки.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 50]
