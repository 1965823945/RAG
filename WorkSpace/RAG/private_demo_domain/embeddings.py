import hashlib
import numpy as np
DIM = 128
def _text_to_seed(text):
    h = hashlib.md5(text.encode('utf-8')).hexdigest()
    return int(h[:8], 16)
def embed_texts(texts):
    vecs = []
    for t in texts:
        seed = _text_to_seed(t)
        rng = np.random.default_rng(seed)
        vecs.append(rng.random(DIM, dtype=np.float32))
    return np.vstack(vecs) if vecs else np.zeros((0, DIM), dtype=np.float32)
