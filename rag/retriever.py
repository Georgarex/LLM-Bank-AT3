from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Load FAISS index and documents
index = faiss.read_index("rag/faiss_index.idx")
with open("rag/docs.pkl", "rb") as f:
    docs = pickle.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_context(query, top_k=3):
    query_vec = embedder.encode([query])
    scores, indices = index.search(query_vec, top_k)
    return "\n".join([docs[i] for i in indices[0]])
