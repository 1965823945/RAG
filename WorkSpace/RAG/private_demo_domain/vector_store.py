import numpy as np
from embeddings import embed_texts, DIM
class VectorStore:
    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.docs = []
        self.vectors = []
    def add_documents(self, texts, metadatas=None):
        if metadatas is None:
            metadatas = [{} for _ in texts]
        self.docs.extend([{"text": t, "meta": m} for t, m in zip(texts, metadatas)])
        if texts:
            vecs = embed_texts(texts).astype(np.float32)
            self.vectors.append(vecs)
    def search(self, query, top_k=3):
        if not self.docs or not self.vectors:
            return []
        qv = embed_texts([query]).astype(np.float32)[0:1]
        all_vecs = np.vstack(self.vectors)
        norms = all_vecs / np.linalg.norm(all_vecs, axis=1, keepdims=True)
        qnorm = qv / np.linalg.norm(qv)
        sims = norms @ qnorm.T
        indices = sims.flatten().argsort()[-top_k:][::-1].tolist()
        return [self.docs[i] for i in indices if i < len(self.docs)]
