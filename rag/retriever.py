import os
import glob
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

# Initialize constants
EMBED_DIM = 384
MODEL_NAME = "paraphrase-MiniLM-L6-v2"
CHUNKS_DIR = "rag/chunks/"

class Retriever:
    """Improved retriever for banking Q&A"""
    
    def __init__(self, chunks_dir=CHUNKS_DIR, model_name=MODEL_NAME):
        self.chunks_dir = chunks_dir
        self.model_name = model_name
        
        # Create directory if it doesn't exist
        os.makedirs(chunks_dir, exist_ok=True)
        
        # Load texts
        self.texts = []
        for path in sorted(glob.glob(f"{self.chunks_dir}/*.txt")):
            with open(path) as f:
                self.texts.append(f.read())
        
        if not self.texts:
            print(f"Warning: No text chunks found in {chunks_dir}")
            # Add a placeholder to avoid empty index
            self.texts = ["No data available. Please run data preparation script."]
                
        # Initialize embedder
        print(f"Loading embedding model: {model_name}")
        self.embedder = SentenceTransformer(model_name)
        self.embed_dim = self.embedder.get_sentence_embedding_dimension()
        
        # Create index
        print("Creating embeddings...")
        self.embs = self.embedder.encode(self.texts, convert_to_numpy=True)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.embed_dim)
        faiss.normalize_L2(self.embs)
        self.index.add(self.embs)
        
        print(f"Retriever initialized with {len(self.texts)} chunks")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve the most relevant chunks for a query"""
        # Embed query
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        
        # Search
        scores, indices = self.index.search(q_emb, top_k)
        
        # Return relevant documents with scores
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            results.append((self.texts[idx], float(score)))
        
        return results
    
    def get_context(self, query: str, top_k: int = 3) -> str:
        """Get the combined context from top results"""
        results = self.retrieve(query, top_k)
        
        # Combine contexts with separators
        contexts = [text for text, _ in results]
        return "\n\n---\n\n".join(contexts)

# Global instance for easy access
_retriever = None

def get_context(query: str, top_k: int = 3) -> str:
    """Legacy function for backward compatibility"""
    global _retriever
    
    # Initialize retriever if needed
    if _retriever is None:
        _retriever = Retriever()
    
    return _retriever.get_context(query, top_k)

if __name__ == "__main__":
    # Test the retriever
    retriever = Retriever()
    query = "How do I reset my password?"
    context = retriever.get_context(query)
    print(f"Query: {query}")
    print(f"Retrieved context length: {len(context)}")