# 🏦 Banking LLM: Fine-tuned + RAG Assistant

A production-ready framework that combines:
- 🔧 Fine-tuned LLM on bank-specific FAQs
- 🔎 Retrieval-Augmented Generation from real bank docs
- 🚀 FastAPI serving endpoint

## 🔍 Example Prompt
```json
POST /ask { "query": "What’s the early repayment penalty on a home loan?" }
```

## 📦 Setup
```bash
pip install -r requirements.txt
python rag/chunker.py     # Embed and index PDF docs
python finetune/train.py  # Fine-tune Mistral
uvicorn app:app --reload  # Launch API
```

## Installing GPT2 local files (over HuggingFace)

#### 1. Install Git LFS (if you haven’t yet - a long file storage version of Git, typically used for model parameters)
```bash
brew install git-lfs      # macOS
git lfs install
```

#### 2. Re-clone (or update) the repo so LFS files download
```bash
rm -rf models/gpt2                                 # remove models/gpt2 if existing
git clone https://huggingface.co/gpt2 models/gpt2  # clone gpt2 parameters into models/gpt2
cd models/gpt2
git lfs pull
```
