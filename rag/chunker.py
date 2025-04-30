# PDF/ text chunking and embedding
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss, pickle

reader = PdfReader("data/bank_policy.pdf")
texts = [page.extract_text() for page in reader.pages]
docs = [chunk.strip() for page in texts for chunk in page.split("\n") if len(chunk) > 40]

model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(docs)

index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

faiss.write_index(index, "rag/faiss_index.idx")
with open("rag/docs.pkl", "wb") as f:
    pickle.dump(docs, f)
