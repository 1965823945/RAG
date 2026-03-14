def chunk_text(text: str, max_len: int = 600, overlap: int = 100):
    if not text:
        return []
    chunks = []
    start = 0
    end = min(len(text), max_len)
    while start < len(text):
        end = min(start + max_len, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks
