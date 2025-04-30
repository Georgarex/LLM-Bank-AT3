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