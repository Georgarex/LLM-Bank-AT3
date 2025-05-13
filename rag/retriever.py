import os
import glob
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional
import threading

# Constants
# EMBED_DIM = 384 # This will be derived from the model
MODEL_NAME = "paraphrase-MiniLM-L6-v2"
CHUNKS_DIR = "rag/chunks/" # Default chunks directory

class Retriever:
    """
    Handles text chunk loading, embedding, and retrieval using FAISS.
    """
    def __init__(self, chunks_dir: str = CHUNKS_DIR, model_name: str = MODEL_NAME):
        self.chunks_dir = chunks_dir
        self.model_name = model_name
        self.texts: List[str] = []
        self.embedder: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None # FAISS index
        self.embed_dim: Optional[int] = None

        print(f"Retriever initializing with chunks_dir='{self.chunks_dir}', model_name='{self.model_name}'")
        self._load_texts()
        self._initialize_embedder_and_index()
        print(f"Retriever initialized successfully with {len(self.texts)} chunks.")

    def _load_texts(self):
        """Loads text chunks from the specified directory."""
        os.makedirs(self.chunks_dir, exist_ok=True)
        
        file_paths = sorted(glob.glob(os.path.join(self.chunks_dir, "*.txt")))
        if not file_paths:
            print(f"WARNING: No text files found in '{self.chunks_dir}'. Adding placeholder text.")
            self.texts = ["No data available. Please ensure your data preparation script has run and populated the chunks directory."]
            return

        for path in file_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self.texts.append(f.read())
            except Exception as e:
                print(f"Error reading file {path}: {e}")
        
        if not self.texts: # Should not happen if file_paths was not empty, but as a safeguard
             print(f"CRITICAL WARNING: Files were found but no texts loaded from '{self.chunks_dir}'. Using placeholder.")
             self.texts = ["No data available despite files being present. Check file contents and permissions."]
        print(f"Loaded {len(self.texts)} text chunks from '{self.chunks_dir}'.")


    def _initialize_embedder_and_index(self):
        """Initializes the sentence embedding model and FAISS index."""
        if not self.texts:
            print("ERROR: Cannot initialize embedder and index because no texts were loaded.")
            return

        print(f"Loading sentence embedding model: '{self.model_name}'...")
        try:
            self.embedder = SentenceTransformer(self.model_name)
            self.embed_dim = self.embedder.get_sentence_embedding_dimension()
            print(f"Embedding model loaded. Embedding dimension: {self.embed_dim}")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load SentenceTransformer model '{self.model_name}': {e}")
            # Potentially raise the error or handle it so the app can start with RAG disabled
            raise RuntimeError(f"Failed to load SentenceTransformer model: {e}") from e


        print("Creating embeddings for text chunks. This may take a moment...")
        try:
            # Ensure embeddings are float32, which FAISS typically expects.
            # SentenceTransformer usually produces float32, but explicit conversion is safe.
            embeddings = self.embedder.encode(self.texts, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to encode texts into embeddings: {e}")
            raise RuntimeError(f"Failed to create embeddings: {e}") from e

        if embeddings.shape[0] == 0:
            print("CRITICAL ERROR: No embeddings were generated. FAISS index cannot be built.")
            # This indicates a serious issue, possibly with the input texts or the embedding model.
            self.texts = ["Embedding generation failed. RAG is non-functional."] # Update texts to reflect state
            # Optionally, create a dummy index or handle this state appropriately
            return

        print(f"Embeddings created. Shape: {embeddings.shape}")

        # Normalize embeddings for cosine similarity search with IndexFlatIP
        # IndexFlatIP calculates dot products. For normalized vectors, dot product is cosine similarity.
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(self.embed_dim)
        self.index.add(embeddings)
        print(f"FAISS index created and populated with {self.index.ntotal} vectors.")

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieves the top_k most relevant chunks for a given query."""
        if self.index is None or self.embedder is None or self.index.ntotal == 0:
            print("WARNING: Retriever.retrieve called but index or embedder is not initialized, or index is empty.")
            return []
        if not query or not query.strip():
            print("WARNING: Retriever.retrieve called with empty query.")
            return []

        try:
            query_embedding = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
            faiss.normalize_L2(query_embedding) # Normalize query embedding as well
        except Exception as e:
            print(f"Error encoding query '{query[:50]}...': {e}")
            return []

        # Ensure top_k is not greater than the number of items in the index
        actual_top_k = min(top_k, self.index.ntotal)
        if actual_top_k == 0: # Should be caught by index.ntotal == 0 above, but good practice
            return []

        try:
            scores, indices = self.index.search(query_embedding, actual_top_k)
        except Exception as e:
            print(f"Error searching FAISS index for query '{query[:50]}...': {e}")
            return []
        
        results = []
        for i in range(actual_top_k):
            idx = indices[0][i]
            score = scores[0][i]
            if idx != -1: # FAISS returns -1 if fewer than k results are found
                results.append((self.texts[idx], float(score)))
        
        return results
    
    def get_context(self, query: str, top_k: int = 3) -> str:
        """Gets the combined context string from top retrieved results."""
        retrieved_docs = self.retrieve(query, top_k)
        if not retrieved_docs:
            return ""
        
        # Combine contexts with a clear separator
        contexts = [text for text, score in retrieved_docs] # score can be used for logging/debugging
        return "\n\n---\n\n".join(contexts)

# --- Thread-safe Singleton instance management for the Retriever ---
_retriever_singleton_instance: Optional[Retriever] = None
_retriever_init_lock = threading.Lock()

def get_retriever() -> Retriever:
    """
    Returns the singleton instance of the Retriever.
    Initializes it on the first call in a thread-safe manner.
    """
    global _retriever_singleton_instance
    if _retriever_singleton_instance is None:
        with _retriever_init_lock: # Ensure only one thread initializes
            if _retriever_singleton_instance is None: # Double-check locking pattern
                print("Global Retriever instance not found. Initializing now...")
                try:
                    _retriever_singleton_instance = Retriever(chunks_dir=CHUNKS_DIR, model_name=MODEL_NAME)
                    print("Global Retriever instance successfully initialized.")
                except Exception as e:
                    print(f"CRITICAL ERROR: Failed to initialize global Retriever instance: {e}")
                    # Depending on desired behavior, you might re-raise, or return a dummy/non-functional retriever
                    # For now, let's re-raise to make it clear initialization failed.
                    raise
    return _retriever_singleton_instance

def get_context(query: str, top_k: int = 3) -> str:
    """Convenience function to get context using the global Retriever instance."""
    try:
        retriever_instance = get_retriever()
        return retriever_instance.get_context(query, top_k)
    except Exception as e:
        print(f"Error in get_context (likely during retriever initialization or usage): {e}")
        # Fallback to empty context if retriever fails catastrophically
        return ""

if __name__ == "__main__":
    print("--- Testing Retriever Module ---")
    
    # Test 1: Initialize and retrieve
    print("\nTest 1: Basic initialization and retrieval...")
    try:
        # This will initialize the global retriever instance.
        retriever = get_retriever() # Use the singleton accessor
        
        test_query_1 = "How do I check my account balance?"
        print(f"Query: \"{test_query_1}\"")
        context_1 = retriever.get_context(test_query_1)
        if context_1:
            print(f"Retrieved context length: {len(context_1)}")
            print(f"Context sample: \"{context_1[:150].replace(os.linesep, ' ')}...\"")
        else:
            print("No context retrieved. Check if chunks directory is populated and model loaded correctly.")

        # Test 2: Empty query
        print("\nTest 2: Empty query string...")
        test_query_2 = ""
        print(f"Query: \"{test_query_2}\"")
        context_2 = retriever.get_context(test_query_2)
        print(f"Retrieved context for empty query (should be empty): '{context_2}'")
        assert context_2 == "", "Context for empty query should be empty"

        # Test 3: Query with no expected matches (if data is specific)
        print("\nTest 3: Query with no expected matches...")
        test_query_3 = "What is the meaning of life in banking?"
        print(f"Query: \"{test_query_3}\"")
        context_3 = retriever.get_context(test_query_3)
        if not context_3:
            print("No context retrieved, as expected for an outlandish query or one not in data.")
        else:
            print(f"Context retrieved (unexpected for this query?): '{context_3[:100]}...'")

    except Exception as e:
        print(f"An error occurred during retriever testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n--- Retriever Module Test Complete ---")