def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    把长文本切成固定大小片段，带重叠
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks
