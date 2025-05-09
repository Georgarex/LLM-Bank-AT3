"""
Prepare data for the RAG system.
Run this script once before starting the API server.
"""

import os
from rag.chunker import chunk

def prepare_data():
    """Prepare data for the RAG system"""
    print("Preparing data for the RAG system...")
    
    # Create necessary directories
    os.makedirs("rag/chunks", exist_ok=True)
    print("Created directory: rag/chunks")
    
    # Run chunking process
    print("Creating chunks from banking Q&A data...")
    chunk(input_path="data/banking_qa.json", out_dir="rag/chunks/")
    
    # Count the number of chunks created
    num_chunks = len([f for f in os.listdir("rag/chunks/") if f.endswith(".txt")])
    print(f"Created {num_chunks} text chunks")
    
    print("\nData preparation complete!")
    print("You can now run the server with: uvicorn app:app --reload")

if __name__ == "__main__":
    prepare_data()