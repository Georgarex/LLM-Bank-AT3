import os
import glob
import faiss
from sentence_transformers import SentenceTransformer

EMBED_DIM = 384
MODEL_NAME = "paraphrase-MiniLM-L6-v2"
CHUNKS_DIR = "rag/chunks/"

# 1. Load all chunk texts
texts = []
for path in sorted(glob.glob(f"{CHUNKS_DIR}/*.txt")):
    with open(path) as f:
        texts.append(f.read())

# 2. Embed and index
embedder = SentenceTransformer(MODEL_NAME)
embs = embedder.encode(texts, convert_to_numpy=True)
index = faiss.IndexFlatIP(EMBED_DIM)
faiss.normalize_L2(embs)
index.add(embs)

def get_context(query: str, top_k: int = 3) -> str:
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    return "\n\n".join(texts[i] for i in I[0])