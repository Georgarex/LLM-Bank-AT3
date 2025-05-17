import json
import os
import re
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def preprocess_text(text):
    """Clean and normalize text"""
    # Replace multiple newlines with single newline
    text = re.sub(r'\n+', '\n', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk(input_path="data/banking_qa.json", out_dir="rag/chunks/"):
    """
    Improved chunking for QA pairs
    
    Args:
        input_path: Path to input JSON file
        out_dir: Directory to save chunks
    """
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Load QA pairs
    with open(input_path) as f:
        qa_pairs = json.load(f)
    
    # Clear existing chunks
    existing_files = [f for f in os.listdir(out_dir) if f.endswith('.txt')]
    for file in existing_files:
        os.remove(os.path.join(out_dir, file))
    
    # Process each QA pair
    for i, pair in enumerate(qa_pairs):
        prompt = pair['prompt'].strip()
        response = pair['response'].strip()
        
        # Create document with clear formatting
        doc = f"Q: {prompt}\nA: {response}"
        
        # For longer QA pairs, we could break them into smaller chunks
        # For simplicity, we'll keep each QA pair together
        with open(os.path.join(out_dir, f"{i:04d}.txt"), "w") as out:
            out.write(doc)
    
    print(f"Created {len(qa_pairs)} chunks in {out_dir}")

# For running directly
if __name__ == "__main__":
    chunk()